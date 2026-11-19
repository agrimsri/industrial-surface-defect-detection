"""
Compute and save continuous test probability maps (classification & segmentation with 4-way TTA)
for Fold 0 and Fold 1 models (Best + SWA snapshots = 4 models).
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

if '.' not in sys.path:
    sys.path.append('.')

from src.models import DualHeadMTFPN

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

def run_tta_for_model(model, imgs):
    """Computes 4-way TTA (Base, H, V, HV) for both heads of a given model."""
    # 1. Base
    cl, sl = model(imgs)
    cp = torch.sigmoid(cl)
    sp = torch.sigmoid(sl)
    del cl, sl
    
    # 2. H-flip
    cl_h, sl_h = model(torch.flip(imgs, dims=[-1]))
    cp = cp + torch.sigmoid(cl_h)
    sp = sp + torch.flip(torch.sigmoid(sl_h), dims=[-1])
    del cl_h, sl_h
    
    # 3. V-flip
    cl_v, sl_v = model(torch.flip(imgs, dims=[-2]))
    cp = cp + torch.sigmoid(cl_v)
    sp = sp + torch.flip(torch.sigmoid(sl_v), dims=[-2])
    del cl_v, sl_v
    
    # 4. HV-flip
    cl_hv, sl_hv = model(torch.flip(imgs, dims=[-2, -1]))
    cp = cp + torch.sigmoid(cl_hv)
    sp = sp + torch.flip(torch.sigmoid(sl_hv), dims=[-2, -1])
    del cl_hv, sl_hv
    
    return cp * 0.25, sp * 0.25

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_paths = [
        'kaggle_output/model_fold_0_best.pth',
        'kaggle_output/model_fold_0_swa.pth',
        'kaggle_output/model_fold_1_best.pth',
        'kaggle_output/model_fold_1_swa.pth'
    ]
    
    sample_sub_path = 'public/sample_submission.csv'
    test_img_dir = 'public/test'
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    sub_template = pd.read_csv(sample_sub_path)
    test_image_ids = sub_template['image_id'].unique().tolist()
    num_test = len(test_image_ids)
    print(f"Total test images: {num_test}")
    
    # Check if models exist
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model checkpoint not found: {mp}")
            
    models = []
    for mp in model_paths:
        print(f"Loading model: {mp}...")
        m = DualHeadMTFPN(backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=False).to(device)
        ckpt = torch.load(mp, map_location=device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models.append(m)
        
    batch_size = 6
    test_ds = TestDefectDataset(test_image_ids, img_dir=test_img_dir)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    all_cls_probs = np.zeros((num_test, 4), dtype=np.float32)
    all_seg_probs = np.zeros((num_test, 4, 128, 800), dtype=np.float16)
    
    t0 = time.time()
    processed_count = 0
    print(f"\n🚀 Computing continuous probability maps for {len(models)} models with 4-way TTA...")
    
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            imgs = batch['image'].to(device)
            B = len(batch['image_id'])
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                batch_cls_list = []
                batch_seg_list = []
                
                for m in models:
                    cp, sp = run_tta_for_model(m, imgs)
                    batch_cls_list.append(cp)
                    batch_seg_list.append(sp)
                    
                cls_probs = torch.stack(batch_cls_list, dim=0).mean(dim=0)
                seg_probs = torch.stack(batch_seg_list, dim=0).mean(dim=0)
                del batch_cls_list, batch_seg_list, imgs
                
            cls_probs_np = cls_probs.cpu().numpy().astype(np.float32)
            seg_probs_np = seg_probs.cpu().numpy().astype(np.float16)
            del cls_probs, seg_probs
            
            start_idx = processed_count
            end_idx = start_idx + B
            all_cls_probs[start_idx:end_idx] = cls_probs_np
            all_seg_probs[start_idx:end_idx] = seg_probs_np
            processed_count += B
            
            del cls_probs_np, seg_probs_np, batch
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(test_loader):
                elapsed = time.time() - t0
                pct = ((b_idx + 1) / len(test_loader)) * 100
                fps = processed_count / elapsed
                print(f"  Processed {processed_count}/{num_test} images ({pct:.1f}%) | {fps:.1f} FPS")
                
    cls_out = os.path.join(save_dir, 'test_cls_probs_folds01.npy')
    seg_out = os.path.join(save_dir, 'test_seg_probs_folds01.npy')
    np.save(cls_out, all_cls_probs)
    np.save(seg_out, all_seg_probs)
    print(f"\n✅ Successfully saved continuous test probabilities for Folds 0 & 1:")
    print(f"   Classification: {cls_out} (shape {all_cls_probs.shape})")
    print(f"   Segmentation:   {seg_out} (shape {all_seg_probs.shape}, {os.path.getsize(seg_out)/(1024*1024):.1f} MB)")

if __name__ == '__main__':
    main()
