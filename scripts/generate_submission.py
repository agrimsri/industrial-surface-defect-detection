import os
import sys
import gc
import json
import time
import argparse
import csv
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if '.' not in sys.path:
    sys.path.append('.')

from src.models import DualHeadMTFPN
from src.dataset import DEFECT_TYPES, TestDefectDataset
from src.postprocess import postprocess_mask, rle_encode

def generate_submission(
    checkpoint_path='checkpoints/production/dualhead_mtfpn_resnet34_best.pth',
    thresholds_path='checkpoints/production/best_thresholds.json',
    output_path='submissions/submission.csv',
    use_tta=True,
    batch_size=12,
    device='cuda'
):
    print("=" * 75)
    print("🚀 PRODUCTION SUBMISSION GENERATOR")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path} | TTA: {use_tta} | Batch Size: {batch_size}")
    print("=" * 75)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load thresholds if available
    cls_thresh = None
    prob_thresh = None
    min_size = None
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'r') as f:
            cfg = json.load(f)
        cls_thresh = cfg.get('cls_thresh', None)
        prob_thresh = cfg.get('prob_thresh', None)
        min_size = cfg.get('min_size', None)
        print(f"Loaded optimal thresholds from {thresholds_path}:")
        if isinstance(cls_thresh, dict):
            for c_k in sorted(cls_thresh.keys()):
                c_idx = int(c_k)
                dname = DEFECT_TYPES[c_idx] if c_idx < len(DEFECT_TYPES) else f'class_{c_k}'
                ct = cls_thresh.get(c_k)
                pt = prob_thresh.get(c_k) if isinstance(prob_thresh, dict) else prob_thresh
                ms = min_size.get(c_k) if isinstance(min_size, dict) else min_size
                print(f"  [{dname:8s}] tau_cls: {ct} | tau_prob: {pt} | min_size: {ms}px")
        else:
            print(f"  tau_cls: {cls_thresh} | tau_prob: {prob_thresh} | min_size: {min_size}")
    else:
        print("Using model default per-class thresholds.")
        
    # 2. Load model
    model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=False).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded weights from {checkpoint_path} (Best recorded score: {ckpt.get('best_score', 'N/A')})")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    model.eval()
    
    # 3. Load sample submission template
    sub_template = pd.read_csv('public/sample_submission.csv')
    test_image_ids = sub_template['image_id'].unique().tolist()
    print(f"Total unique test images to predict: {len(test_image_ids)}")
    
    test_ds = TestDefectDataset(test_image_ids, img_dir='public/test')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    # 4. Predict in batches with immediate RLE encoding (avoids storing 1.2 GB full mask arrays in RAM)
    rle_dict = {}  # (image_id, class_idx) -> rle string
    
    t0 = time.time()
    processed_count = 0
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            imgs = batch['image'].to(device)
            img_ids = batch['image_id']
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                cls_logits, seg_logits = model(imgs)
                cls_probs = torch.sigmoid(cls_logits)
                seg_probs = torch.sigmoid(seg_logits)
                
                if use_tta:
                    imgs_flip = torch.flip(imgs, dims=[-1])
                    cls_logits_f, seg_logits_f = model(imgs_flip)
                    cls_probs_f = torch.sigmoid(cls_logits_f)
                    seg_probs_f = torch.flip(torch.sigmoid(seg_logits_f), dims=[-1])
                    
                    cls_probs = 0.5 * (cls_probs + cls_probs_f)
                    seg_probs = 0.5 * (seg_probs + seg_probs_f)
                    
            cls_probs_np = cls_probs.cpu().numpy()
            seg_probs_np = seg_probs.cpu().numpy()
            
            B, C, H, W = seg_probs_np.shape
            for i in range(B):
                img_id = img_ids[i]
                for c in range(4):
                    clean_mask = postprocess_mask(
                        seg_probs_np[i, c],
                        cls_prob=cls_probs_np[i, c],
                        cls_idx=c,
                        cls_thresh=cls_thresh,
                        prob_thresh=prob_thresh,
                        min_size=min_size
                    )
                    rle = rle_encode(clean_mask)
                    rle_dict[(img_id, c)] = rle
                processed_count += 1
                
            del imgs, cls_logits, seg_logits, cls_probs, seg_probs, cls_probs_np, seg_probs_np, batch
            if use_tta:
                del imgs_flip, cls_logits_f, seg_logits_f, cls_probs_f, seg_probs_f
                
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(test_loader):
                elapsed = time.time() - t0
                pct = ((b_idx + 1) / len(test_loader)) * 100
                fps = processed_count / elapsed
                print(f"  Processed {processed_count}/{len(test_image_ids)} images ({pct:.1f}%) | Speed: {fps:.1f} FPS")
                
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 5. Format rows into DataFrame matching template exactly
    print("Formatting submission rows matching sample_submission.csv...")
    rows = []
    for idx, row in sub_template.iterrows():
        row_id = row['ID']
        img_id = row['image_id']
        d_type = row['defect_type']
        c_idx = DEFECT_TYPES.index(d_type)
        
        rle = rle_dict.get((img_id, c_idx), '')
        formatted_rle = str(rle).strip() if str(rle).strip() != '' else ' '
        rows.append({
            'ID': row_id,
            'image_id': img_id,
            'defect_type': d_type,
            'mask_rle': formatted_rle
        })
        
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved submission to {output_path} (quoted with ' ' for empty masks)")
    
    # 6. Verification
    print("\n🔍 SUBMISSION AUDIT & INTEGRITY CHECK:")
    print(f"  Row count: {len(out_df)} (Target: 12000) -> {'PASS' if len(out_df) == 12000 else 'FAIL'}")
    print(f"  Columns: {list(out_df.columns)} -> {'PASS' if list(out_df.columns) == ['ID', 'image_id', 'defect_type', 'mask_rle'] else 'FAIL'}")
    print(f"  Null count: {out_df.isnull().sum().sum()} -> {'PASS' if out_df.isnull().sum().sum() == 0 else 'FAIL'}")
    
    empty_masks = (out_df['mask_rle'].str.strip() == '').sum()
    empty_pct = (empty_masks / len(out_df)) * 100
    pos_masks = len(out_df) - empty_masks
    print(f"  Empty / Clean Masks: {empty_masks:,} ({empty_pct:.2f}%)")
    print(f"  Positive Masks: {pos_masks:,} ({100 - empty_pct:.2f}%)")
    
    print("\n  Per-Class Defect Counts:")
    for d_type in DEFECT_TYPES:
        c_count = (out_df[out_df['defect_type'] == d_type]['mask_rle'].str.strip() != '').sum()
        print(f"    - {d_type:8s}: {c_count:5d} masks ({c_count/3000*100:.2f}% of test images)")
        assert c_count > 0, f"Error: Defect class {d_type} has 0 predicted instances!"
    print("  All 4 defect categories have active, non-zero predictions.")
    
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/production/dualhead_mtfpn_resnet34_best.pth')
    parser.add_argument('--thresholds', default='checkpoints/production/best_thresholds.json')
    parser.add_argument('--output', default='submissions/submission.csv')
    parser.add_argument('--no-tta', dest='use_tta', action='store_false')
    parser.set_defaults(use_tta=True)
    args = parser.parse_args()
    
    generate_submission(
        checkpoint_path=args.checkpoint,
        thresholds_path=args.thresholds,
        output_path=args.output,
        use_tta=args.use_tta
    )
