import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if '.' not in sys.path:
    sys.path.append('.')

from src.dataset import prepare_image_dataframe, DefectDataset
from src.models import DualHeadMTFPN
from src.metrics import StreamingMetricMeter
from src.postprocess import postprocess_mask

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

def run_cache():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    val_fold_df = df[df['fold'] == 0].copy()
    val_eval_df = val_fold_df.sample(n=min(1000, len(val_fold_df)), random_state=42).reset_index(drop=True)
    
    val_ds = DefectDataset(val_eval_df, img_dir='public/train', is_train=False, augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=False)
    
    model = DualHeadMTFPN(backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=False).to(device)
    ckpt = torch.load('kaggle_output/model_best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    all_cls_probs = []
    all_seg_probs = []
    all_targets = []
    
    print("Pre-computing predictions with 4-way TTA on 1000 validation images...")
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            imgs = batch['image'].to(device)
            targets = (batch['mask'].numpy() > 0).astype(np.uint8)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 1. Base
                cl, sl = model(imgs)
                cp = torch.sigmoid(cl)
                sp = torch.sigmoid(sl)
                
                # 2. H-flip
                cl_h, sl_h = model(torch.flip(imgs, dims=[-1]))
                cp_h = torch.sigmoid(cl_h)
                sp_h = torch.flip(torch.sigmoid(sl_h), dims=[-1])
                
                # 3. V-flip
                cl_v, sl_v = model(torch.flip(imgs, dims=[-2]))
                cp_v = torch.sigmoid(cl_v)
                sp_v = torch.flip(torch.sigmoid(sl_v), dims=[-2])
                
                # 4. HV-flip
                cl_hv, sl_hv = model(torch.flip(imgs, dims=[-2, -1]))
                cp_hv = torch.sigmoid(cl_hv)
                sp_hv = torch.flip(torch.sigmoid(sl_hv), dims=[-2, -1])
                
                cls_probs = 0.25 * (cp + cp_h + cp_v + cp_hv)
                seg_probs = 0.25 * (sp + sp_h + sp_v + sp_hv)
                
            all_cls_probs.append(cls_probs.cpu().numpy().astype(np.float32))
            all_seg_probs.append(seg_probs.cpu().numpy().astype(np.float16))
            all_targets.append(targets)
            
            if (batch_idx + 1) % 25 == 0:
                print(f"  Processed {(batch_idx + 1) * 8}/1000 images ({time.time() - t0:.1f}s)")
                
    all_cls_probs = np.concatenate(all_cls_probs, axis=0)
    all_seg_probs = np.concatenate(all_seg_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    os.makedirs('checkpoints', exist_ok=True)
    np.savez_compressed('checkpoints/val_tta_cache.npz',
                        cls_probs=all_cls_probs,
                        seg_probs=all_seg_probs,
                        targets=all_targets)
    print(f"Cached validation predictions in checkpoints/val_tta_cache.npz ({time.time() - t0:.1f}s)")

if __name__ == '__main__':
    run_cache()
