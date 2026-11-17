import os
import sys
import gc
import json
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
    print("=" * 70)
    print("🎯 OPTIMIZING TWO-STAGE POST-PROCESSING THRESHOLDS")
    print(f"Loading checkpoint: {checkpoint_path}")
    print("=" * 70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    val_fold_df = df[df['fold'] == fold].sample(n=min(val_samples, len(df[df['fold'] == fold])), random_state=42).reset_index(drop=True)
    val_ds = DefectDataset(val_fold_df, img_dir='public/train', is_train=False, augment=False)
    # Memory hygiene: num_workers=0, pin_memory=False
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    
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
            
    all_cls_probs = np.concatenate(all_cls_probs, axis=0)
    all_seg_probs = np.concatenate(all_seg_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    N, C, H, W = all_seg_probs.shape
    print(f"Validation cache ready: {N} samples, {C} classes. Memory optimized (FP16 seg probs, uint8 targets).")
    
    # Grid search over classification thresholds tau_cls and min_sizes
    candidate_cls_thresh = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60]
    candidate_min_sizes = [30, 50, 75, 100]
    
    best_score = -1.0
    best_config = {}
    
    print("\nRunning grid search across gating thresholds:")
    for ct in candidate_cls_thresh:
        for ms in candidate_min_sizes:
            meter = StreamingMetricMeter(num_classes=C)
            chunk_size = 50
            for start_idx in range(0, N, chunk_size):
                end_idx = min(start_idx + chunk_size, N)
                chunk_B = end_idx - start_idx
                chunk_preds = np.zeros((chunk_B, C, H, W), dtype=np.uint8)
                for b_i in range(chunk_B):
                    idx = start_idx + b_i
                    for c in range(C):
                        chunk_preds[b_i, c] = postprocess_mask(
                            all_seg_probs[idx, c].astype(np.float32),
                            cls_prob=all_cls_probs[idx, c],
                            cls_idx=c,
                            cls_thresh=ct,
                            prob_thresh=0.50,
                            min_size=ms
                        )
                meter.update(chunk_preds, all_targets[start_idx:end_idx])
                del chunk_preds
                
            metrics = meter.compute()
            score = metrics['final_score']
            print(f"  tau_cls: {ct:.2f} | min_size: {ms:3d} --> Score: {score:.5f} (Dice: {metrics['s_dice']:.4f}, FP: {metrics['s_fp']:.4f}, Empty: {metrics['s_empty']:.4f})")
            
            if score > best_score:
                best_score = score
                best_config = {
                    'cls_thresh': ct,
                    'prob_thresh': 0.50,
                    'min_size': ms,
                    'best_val_score': score,
                    'val_dice': metrics['s_dice'],
                    'val_fp_score': metrics['s_fp'],
                    'val_empty_acc': metrics['s_empty'],
                    'per_class_dice': metrics['per_class_dice']
                }
                
    print("\n" + "=" * 70)
    print(f"🏆 BEST POST-PROCESSING CONFIGURATION:")
    print(f"  Classification Gate tau_cls: {best_config['cls_thresh']}")
    print(f"  Probability Threshold tau_prob: {best_config['prob_thresh']}")
    print(f"  Min Component Size: {best_config['min_size']} px")
    print(f"  Optimal Validation Score: {best_config['best_val_score']:.5f}")
    print("=" * 70)
    
    out_json = 'checkpoints/production/best_thresholds.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved optimal configuration to {out_json}")
    
    del all_cls_probs, all_seg_probs, all_targets
    gc.collect()
    return best_config

if __name__ == '__main__':
    tune_postprocessing_thresholds()
