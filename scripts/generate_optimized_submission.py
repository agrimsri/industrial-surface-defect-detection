import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if '.' not in sys.path:
    sys.path.append('.')

from src.models import DualHeadMTFPN
from src.postprocess import postprocess_mask, rle_encode

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

class TestDefectDataset(Dataset):
    def __init__(self, image_ids, img_dir):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f'{image_id}.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Test image not found: {img_path}")
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        return {'image': img_tensor, 'image_id': image_id}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sample_sub_path = 'public/sample_submission.csv'
    test_img_dir = 'public/test'
    model_path = 'kaggle_output/model_best.pth'
    output_path = 'submissions/submission_optimized.csv'
    
    # Calibrated optimal thresholds
    thresholds_cfg = {
        'cls_thresh': {'0': 0.30, '1': 0.40, '2': 0.35, '3': 0.25},
        'prob_thresh': {'0': 0.45, '1': 0.40, '2': 0.50, '3': 0.45},
        'min_size': {'0': 40, '1': 200, '2': 60, '3': 200}
    }
    
    print("\nOptimal Threshold Configuration:")
    print(json.dumps(thresholds_cfg, indent=2))
    
    sub_template = pd.read_csv(sample_sub_path)
    test_image_ids = sub_template['image_id'].unique().tolist()
    print(f"Total test images: {len(test_image_ids)}")
    
    # Batch size 6 ensures rock-solid memory stability on 4GB laptop GPU
    batch_size = 6
    test_ds = TestDefectDataset(test_image_ids, img_dir=test_img_dir)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    model = DualHeadMTFPN(backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    rle_dict = {}
    all_test_cls_probs = []
    
    t0 = time.time()
    processed_count = 0
    
    print("Generating predictions with memory-efficient 4-way TTA...")
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            imgs = batch['image'].to(device)
            img_ids = batch['image_id']
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 1. Base
                cl, sl = model(imgs)
                cls_probs = torch.sigmoid(cl)
                seg_probs = torch.sigmoid(sl)
                del cl, sl
                
                # 2. H-flip
                cl_h, sl_h = model(torch.flip(imgs, dims=[-1]))
                cls_probs = cls_probs + torch.sigmoid(cl_h)
                seg_probs = seg_probs + torch.flip(torch.sigmoid(sl_h), dims=[-1])
                del cl_h, sl_h
                
                # 3. V-flip
                cl_v, sl_v = model(torch.flip(imgs, dims=[-2]))
                cls_probs = cls_probs + torch.sigmoid(cl_v)
                seg_probs = seg_probs + torch.flip(torch.sigmoid(sl_v), dims=[-2])
                del cl_v, sl_v
                
                # 4. HV-flip
                cl_hv, sl_hv = model(torch.flip(imgs, dims=[-2, -1]))
                cls_probs = cls_probs + torch.sigmoid(cl_hv)
                seg_probs = seg_probs + torch.flip(torch.sigmoid(sl_hv), dims=[-2, -1])
                del cl_hv, sl_hv, imgs
                
                cls_probs = cls_probs * 0.25
                seg_probs = seg_probs * 0.25
                
            cls_probs_np = cls_probs.cpu().numpy().astype(np.float32)
            seg_probs_np = seg_probs.cpu().numpy().astype(np.float16)
            del cls_probs, seg_probs
            
            all_test_cls_probs.append(cls_probs_np)
            
            B = len(img_ids)
            for i in range(B):
                img_id = img_ids[i]
                for c in range(4):
                    clean_mask = postprocess_mask(
                        seg_probs_np[i, c].astype(np.float32),
                        cls_prob=cls_probs_np[i, c],
                        cls_idx=c,
                        cls_thresh=thresholds_cfg['cls_thresh'],
                        prob_thresh=thresholds_cfg['prob_thresh'],
                        min_size=thresholds_cfg['min_size']
                    )
                    rle = rle_encode(clean_mask)
                    rle_dict[(img_id, c)] = rle
                processed_count += 1
                
            del seg_probs_np, cls_probs_np
            if (b_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                elapsed = time.time() - t0
                pct = ((b_idx + 1) / len(test_loader)) * 100
                fps = processed_count / elapsed
                print(f"  Processed {processed_count}/{len(test_image_ids)} images ({pct:.1f}%) | {fps:.1f} FPS ({elapsed:.1f}s)")
                
    # Save test classification probabilities
    all_test_cls_probs = np.concatenate(all_test_cls_probs, axis=0)
    np.save('checkpoints/test_cls_probs.npy', all_test_cls_probs)
    print("Saved checkpoints/test_cls_probs.npy")
    
    # Save submission
    rows = []
    for _, row in sub_template.iterrows():
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
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved submission to {output_path}")
    
    # Audit
    print("\n🔍 SUBMISSION AUDIT & INTEGRITY CHECK:")
    print(f"  Row count: {len(out_df)} (Target: 12000)")
    print(f"  Columns: {list(out_df.columns)}")
    print(f"  Null count: {out_df.isnull().sum().sum()}")
    
    empty_masks = (out_df['mask_rle'].str.strip() == '').sum()
    empty_pct = (empty_masks / len(out_df)) * 100
    pos_masks = len(out_df) - empty_masks
    print(f"  Empty Masks: {empty_masks:,} ({empty_pct:.2f}%)")
    print(f"  Positive Masks: {pos_masks:,} ({100 - empty_pct:.2f}%)")
    
    print("\n  Per-Class Defect Counts:")
    for d_type in DEFECT_TYPES:
        c_count = (out_df[out_df['defect_type'] == d_type]['mask_rle'].str.strip() != '').sum()
        print(f"    - {d_type:8s}: {c_count:5d} masks ({c_count/3000*100:.2f}% of test images)")

if __name__ == '__main__':
    main()
