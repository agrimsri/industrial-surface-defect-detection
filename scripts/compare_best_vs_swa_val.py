import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if '.' not in sys.path:
    sys.path.append('.')

from src.dataset import prepare_image_dataframe, DefectDataset
from src.models import DualHeadMTFPN
from src.postprocess import postprocess_mask

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    val_fold_df = df[df['fold'] == 0].copy()
    val_eval_df = val_fold_df.sample(n=min(1000, len(val_fold_df)), random_state=42).reset_index(drop=True)
    val_ds = DefectDataset(val_eval_df, img_dir='public/train', is_train=False, augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)
    
    # Load model_swa
    model_swa = DualHeadMTFPN(backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=False).to(device)
    ckpt = torch.load('kaggle_output/model_swa.pth', map_location=device)
    model_swa.load_state_dict(ckpt['model_state_dict'])
    model_swa.eval()
    
    swa_cls_probs = []
    swa_seg_probs = []
    
    print("Inference on model_swa with 4-way TTA...")
    t0 = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            imgs = batch['image'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 1. Base
                cl, sl = model_swa(imgs)
                cp, sp = torch.sigmoid(cl), torch.sigmoid(sl)
                # 2. H
                cl_h, sl_h = model_swa(torch.flip(imgs, dims=[-1]))
                cp_h, sp_h = torch.sigmoid(cl_h), torch.flip(torch.sigmoid(sl_h), dims=[-1])
                # 3. V
                cl_v, sl_v = model_swa(torch.flip(imgs, dims=[-2]))
                cp_v, sp_v = torch.sigmoid(cl_v), torch.flip(torch.sigmoid(sl_v), dims=[-2])
                # 4. HV
                cl_hv, sl_hv = model_swa(torch.flip(imgs, dims=[-2, -1]))
                cp_hv, sp_hv = torch.sigmoid(cl_hv), torch.flip(torch.sigmoid(sl_hv), dims=[-2, -1])
                
                cls_p = 0.25 * (cp + cp_h + cp_v + cp_hv)
                seg_p = 0.25 * (sp + sp_h + sp_v + sp_hv)
                
            swa_cls_probs.append(cls_p.cpu().numpy().astype(np.float32))
            swa_seg_probs.append(seg_p.cpu().numpy().astype(np.float16))
            
    swa_cls_probs = np.concatenate(swa_cls_probs, axis=0)
    swa_seg_probs = np.concatenate(swa_seg_probs, axis=0)
    print(f"Finished model_swa TTA in {time.time() - t0:.1f}s")
    
    # Load model_best cache
    best_data = np.load('checkpoints/val_tta_cache.npz')
    best_cls = best_data['cls_probs']
    best_seg = best_data['seg_probs']
    targets = best_data['targets']
    N = len(targets)
    
    # Evaluate 3 candidates: Best, SWA, Blend (50/50)
    blend_cls = 0.5 * (best_cls + swa_cls_probs)
    blend_seg = 0.5 * (best_seg.astype(np.float32) + swa_seg_probs.astype(np.float32))
    
    # Optimal thresholds:
    configs = [
        ('defect_a', 0, 0.30, 0.45, 40),
        ('defect_b', 1, 0.40, 0.40, 200),
        ('defect_c', 2, 0.35, 0.50, 60),
        ('defect_d', 3, 0.25, 0.45, 200)
    ]
    
    models = [
        ('model_best', best_cls, best_seg),
        ('model_swa', swa_cls_probs, swa_seg_probs),
        ('model_blend_50_50', blend_cls, blend_seg)
    ]
    
    for m_name, c_probs, s_probs in models:
        total_score = 0
        print(f"\n--- Evaluating {m_name} ---")
        for d_name, c, ct, pt, ms in configs:
            c_cls = c_probs[:, c]
            c_seg = s_probs[:, c]
            c_tgt = targets[:, c]
            t_sums = c_tgt.reshape(N, -1).sum(axis=1)
            pos_idx = np.where(t_sums > 0)[0]
            neg_idx = np.where(t_sums == 0)[0]
            
            pos_dices = []
            neg_fp = []
            neg_emp = []
            pred_cnt = 0
            
            for idx in pos_idx:
                p_cls = c_cls[idx]
                if p_cls < ct:
                    pos_dices.append(0.0)
                else:
                    m = postprocess_mask(c_seg[idx].astype(np.float32), cls_prob=p_cls, cls_idx=c, cls_thresh=ct, prob_thresh=pt, min_size=ms)
                    ps = m.sum()
                    if ps == 0:
                        pos_dices.append(0.0)
                    else:
                        pred_cnt += 1
                        inter = np.logical_and(m, c_tgt[idx]).sum()
                        pos_dices.append(float(2.0 * inter / (ps + t_sums[idx])))
                        
            for idx in neg_idx:
                p_cls = c_cls[idx]
                if p_cls < ct:
                    neg_fp.append(1.0); neg_emp.append(1.0)
                else:
                    m = postprocess_mask(c_seg[idx].astype(np.float32), cls_prob=p_cls, cls_idx=c, cls_thresh=ct, prob_thresh=pt, min_size=ms)
                    ps = m.sum()
                    if ps == 0:
                        neg_fp.append(1.0); neg_emp.append(1.0)
                    else:
                        pred_cnt += 1
                        pen = min(1.0, float(ps) / 100.0)
                        neg_fp.append(max(0.0, 1.0 - pen))
                        neg_emp.append(0.0)
                        
            sd = float(np.mean(pos_dices))
            sfp = float(np.mean(neg_fp))
            se = float(np.mean(neg_emp))
            sc = 0.7*sd + 0.2*sfp + 0.1*se
            total_score += 0.25 * sc
            print(f"  [{d_name:8s}] Score={sc:.5f} (Dice={sd:.4f}, FP={sfp:.4f}, Emp={se:.4f}) | Pred={pred_cnt}")
        print(f"  >>> Overall Score for {m_name}: {total_score:.5f}")

if __name__ == '__main__':
    main()
