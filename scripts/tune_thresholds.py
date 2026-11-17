import os
import sys
import gc
import json
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

if '.' not in sys.path:
    sys.path.append('.')

from src.dataset import prepare_image_dataframe, DefectDataset, DEFECT_TYPES
from src.models import DualHeadMTFPN
from src.metrics import StreamingMetricMeter, compute_competition_metric
from src.postprocess import postprocess_mask

def tune_postprocessing_thresholds(
    checkpoint_path='checkpoints/production/dualhead_mtfpn_resnet34_best.pth',
    val_samples=1000,
    fold=0,
    device='cuda'
):
    print("=" * 75)
    print("🎯 OPTIMIZING TWO-STAGE POST-PROCESSING THRESHOLDS (PER-CLASS CALIBRATION)")
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 75)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    val_fold_df = df[df['fold'] == fold].sample(n=min(val_samples, len(df[df['fold'] == fold])), random_state=42).reset_index(drop=True)
    val_ds = DefectDataset(val_fold_df, img_dir='public/train', is_train=False, augment=False)
    # Memory hygiene: num_workers=0, pin_memory=False
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Predicting probabilities on validation set ({len(val_fold_df)} images)...")
    all_cls_probs = []
    all_seg_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            # Memory optimization: uint8 targets save 75% memory
            targets = (batch['mask'].numpy() > 0).astype(np.uint8)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                cls_logits, seg_logits = model(images)
                
            cls_p = torch.sigmoid(cls_logits).cpu().numpy().astype(np.float32)
            # Memory optimization: float16 probabilities save 50% memory
            seg_p = torch.sigmoid(seg_logits).cpu().numpy().astype(np.float16)
            
            all_cls_probs.append(cls_p)
            all_seg_probs.append(seg_p)
            all_targets.append(targets)
            
            del images, targets, cls_logits, seg_logits, cls_p, seg_p, batch
            
    all_cls_probs = np.concatenate(all_cls_probs, axis=0)  # (N, 4)
    all_seg_probs = np.concatenate(all_seg_probs, axis=0)  # (N, 4, 128, 800)
    all_targets = np.concatenate(all_targets, axis=0)      # (N, 4, 128, 800)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    N, C, H, W = all_seg_probs.shape
    print(f"Validation cache ready: {N} samples, {C} classes. Memory optimized (FP16 seg probs, uint8 targets).")
    
    # Class-specific candidate search grids tailored to each defect morphology
    class_search_grids = {
        0: {  # defect_a: small localized pits (mean area 748 px)
            'cls': [0.10, 0.20, 0.30, 0.40],
            'prob': [0.0001, 0.0003, 0.0005, 0.001, 0.005, 0.01],
            'min_size': [20, 30, 50, 75]
        },
        1: {  # defect_b: faint vertical scratches (rare, 313 positive instances)
            'cls': [0.05, 0.10, 0.15, 0.20],
            'prob': [0.00005, 0.0001, 0.0003, 0.0005, 0.001],
            'min_size': [15, 25, 40, 60]
        },
        2: {  # defect_c: large irregular flake patches (mean area 5558 px)
            'cls': [0.30, 0.35, 0.40, 0.45, 0.50],
            'prob': [0.35, 0.40, 0.50, 0.55],
            'min_size': [40, 60, 75, 100]
        },
        3: {  # defect_d: continuous rolling abrasion bands (mean area 7150 px)
            'cls': [0.20, 0.25, 0.30, 0.35, 0.40],
            'prob': [0.35, 0.40, 0.45, 0.50],
            'min_size': [50, 75, 100, 120]
        }
    }
    
    best_per_class = {}
    print("\n🔍 Running independent per-class threshold optimization:")
    
    for c, dname in enumerate(DEFECT_TYPES):
        t0 = time.time()
        c_targets = all_targets[:, c]  # (N, 128, 800)
        c_cls = all_cls_probs[:, c]    # (N,)
        c_seg = all_seg_probs[:, c]    # (N, 128, 800)
        
        t_sums = c_targets.reshape(N, -1).sum(axis=1)
        pos_indices = np.where(t_sums > 0)[0]
        neg_indices = np.where(t_sums == 0)[0]
        
        grid = class_search_grids[c]
        best_c_score = -1.0
        best_c_cfg = None
        
        for ct in grid['cls']:
            for pt in grid['prob']:
                for ms in grid['min_size']:
                    pos_dices = []
                    neg_fp = []
                    neg_emp = []
                    
                    # Positive instances
                    for idx in pos_indices:
                        t_sum = t_sums[idx]
                        p_cls = c_cls[idx]
                        if p_cls < ct:
                            pos_dices.append(0.0)
                        else:
                            mask = postprocess_mask(
                                c_seg[idx].astype(np.float32),
                                cls_prob=p_cls,
                                cls_idx=c,
                                cls_thresh=ct,
                                prob_thresh=pt,
                                min_size=ms
                            )
                            p_sum = mask.sum()
                            if p_sum == 0:
                                pos_dices.append(0.0)
                            else:
                                inter = np.logical_and(mask, c_targets[idx]).sum()
                                pos_dices.append(float(2.0 * inter / (p_sum + t_sum)))
                                
                    # Negative instances
                    for idx in neg_indices:
                        p_cls = c_cls[idx]
                        if p_cls < ct:
                            neg_fp.append(1.0)
                            neg_emp.append(1.0)
                        else:
                            mask = postprocess_mask(
                                c_seg[idx].astype(np.float32),
                                cls_prob=p_cls,
                                cls_idx=c,
                                cls_thresh=ct,
                                prob_thresh=pt,
                                min_size=ms
                            )
                            p_sum = mask.sum()
                            if p_sum == 0:
                                neg_fp.append(1.0)
                                neg_emp.append(1.0)
                            else:
                                penalty = min(1.0, float(p_sum) / 500.0)
                                neg_fp.append(max(0.0, 1.0 - penalty))
                                neg_emp.append(0.0)
                                
                    s_dice = float(np.mean(pos_dices)) if pos_dices else 0.0
                    s_fp = float(np.mean(neg_fp)) if neg_fp else 1.0
                    s_emp = float(np.mean(neg_emp)) if neg_emp else 1.0
                    score = 0.70 * s_dice + 0.20 * s_fp + 0.10 * s_emp
                    
                    if score > best_c_score:
                        best_c_score = score
                        best_c_cfg = {
                            'cls_thresh': ct,
                            'prob_thresh': pt,
                            'min_size': ms,
                            'score': score,
                            's_dice': s_dice,
                            's_fp': s_fp,
                            's_empty': s_emp
                        }
                        
        elapsed = time.time() - t0
        best_per_class[c] = best_c_cfg
        print(f"  Class {c} ({dname:8s}) [{elapsed:4.1f}s] -> Optimal Score: {best_c_cfg['score']:.5f} | "
              f"Dice: {best_c_cfg['s_dice']:.4f}, FP: {best_c_cfg['s_fp']:.4f}, Emp: {best_c_cfg['s_empty']:.4f} | "
              f"tau_cls={best_c_cfg['cls_thresh']}, tau_prob={best_c_cfg['prob_thresh']}, min_size={best_c_cfg['min_size']}px")
        
    # Evaluate full composite competition metric with optimal per-class thresholds
    print("\nEvaluating overall validation composite metric with optimal per-class parameters...")
    meter_all = StreamingMetricMeter(num_classes=C)
    chunk_size = 50
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk_B = end_idx - start_idx
        chunk_preds = np.zeros((chunk_B, C, H, W), dtype=np.uint8)
        for b_i in range(chunk_B):
            idx = start_idx + b_i
            for c in range(C):
                cfg = best_per_class[c]
                chunk_preds[b_i, c] = postprocess_mask(
                    all_seg_probs[idx, c].astype(np.float32),
                    cls_prob=all_cls_probs[idx, c],
                    cls_idx=c,
                    cls_thresh=cfg['cls_thresh'],
                    prob_thresh=cfg['prob_thresh'],
                    min_size=cfg['min_size']
                )
        meter_all.update(chunk_preds, all_targets[start_idx:end_idx])
        del chunk_preds
        
    full_metrics = meter_all.compute()
    
    best_config = {
        'best_val_score': full_metrics['final_score'],
        'val_dice': full_metrics['s_dice'],
        'val_fp_score': full_metrics['s_fp'],
        'val_empty_acc': full_metrics['s_empty'],
        'cls_thresh': {str(c): best_per_class[c]['cls_thresh'] for c in range(C)},
        'prob_thresh': {str(c): best_per_class[c]['prob_thresh'] for c in range(C)},
        'min_size': {str(c): best_per_class[c]['min_size'] for c in range(C)},
        'per_class': {
            str(c): {
                'defect_type': DEFECT_TYPES[c],
                'cls_thresh': best_per_class[c]['cls_thresh'],
                'prob_thresh': best_per_class[c]['prob_thresh'],
                'min_size': best_per_class[c]['min_size'],
                'val_score': best_per_class[c]['score'],
                'val_dice': best_per_class[c]['s_dice'],
                'val_fp': best_per_class[c]['s_fp'],
                'val_empty': best_per_class[c]['s_empty']
            } for c in range(C)
        },
        'per_class_dice': {str(c): full_metrics['per_class_dice'][c] for c in range(C)}
    }
    
    print("\n" + "=" * 75)
    print("🏆 OPTIMAL MULTI-CLASS POST-PROCESSING CONFIGURATION:")
    print(f"  Overall Validation Score: {best_config['best_val_score']:.5f}")
    print(f"  Foreground Dice Score:   {best_config['val_dice']:.4f} (70% weight)")
    print(f"  False Positive Score:   {best_config['val_fp_score']:.4f} (20% weight)")
    print(f"  Empty Accuracy Score:   {best_config['val_empty_acc']:.4f} (10% weight)")
    for c in range(C):
        info = best_config['per_class'][str(c)]
        print(f"  - {info['defect_type']:8s}: tau_cls={info['cls_thresh']}, tau_prob={info['prob_thresh']}, min_size={info['min_size']}px | Dice={info['val_dice']:.4f}")
    print("=" * 75)
    
    out_json = 'checkpoints/production/best_thresholds.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved optimal configuration to {out_json}")
    
    del all_cls_probs, all_seg_probs, all_targets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_config

if __name__ == '__main__':
    tune_postprocessing_thresholds()
