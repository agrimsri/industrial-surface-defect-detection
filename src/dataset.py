import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']
DEFECT_TO_IDX = {d: i for i, d in enumerate(DEFECT_TYPES)}
IDX_TO_DEFECT = {i: d for i, d in enumerate(DEFECT_TYPES)}

def rle_decode(rle_str, shape=(128, 800)):
    """
    Decodes Run-Length Encoded string to binary mask.
    Index base: 1-indexed, Column-major (Fortran order).
    """
    if pd.isna(rle_str) or not str(rle_str).strip():
        return np.zeros(shape, dtype=np.uint8)
    s = str(rle_str).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def rle_encode(mask):
    """
    Encodes binary mask (128, 800) to Run-Length Encoded string.
    Index base: 1-indexed, Column-major (Fortran order).
    """
    dots = mask.flatten(order='F')
    if not np.any(dots):
        return ''
    runs = np.where(dots[1:] != dots[:-1])[0] + 2
    runs = np.concatenate([
        [1] if dots[0] else np.empty(0, dtype=int),
        runs,
        [len(dots) + 1] if dots[-1] else np.empty(0, dtype=int)
    ]).astype(int)
    starts = runs[::2]
    lengths = runs[1::2] - starts
    return ' '.join(f'{int(s)} {int(l)}' for s, l in zip(starts, lengths))

def prepare_image_dataframe(csv_path='public/train.csv', n_splits=5, seed=42):
    """
    Pivots long-format train.csv to 1 row per unique image with 15-state stratification.
    """
    df = pd.read_csv(csv_path)
    # Pivot so each row is an image
    pivoted = df.pivot(index='image_id', columns='defect_type', values='mask_rle').reset_index()
    pivoted.columns.name = None
    
    # Compute presence (binary label) for each defect type
    for d in DEFECT_TYPES:
        if d not in pivoted.columns:
            pivoted[d] = ''
        pivoted[f'{d}_has'] = pivoted[d].notna() & (pivoted[d].str.strip() != '')
    
    # Compute the 15 discrete defect combination states for exact StratifiedKFold
    def get_strat_label(row):
        active = [str(i) for i, d in enumerate(DEFECT_TYPES) if row[f'{d}_has']]
        return '_'.join(active) if active else 'none'
    
    pivoted['strat_label'] = pivoted.apply(get_strat_label, axis=1)
    
    # 15-state Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pivoted['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(pivoted, pivoted['strat_label'])):
        pivoted.loc[val_idx, 'fold'] = fold
        
    return pivoted

class DefectDataset(Dataset):
    """
    High-performance Grayscale Surface Defect Dataset.
    Loads 1-channel grayscale directly (saving 66.7% memory & PCIe bandwidth).
    Returns:
      image: Tensor (1, 128, 800), normalized to [0, 1]
      cls_labels: Tensor (4,) float32 {0, 1}
      seg_masks: Tensor (4, 128, 800) float32 {0, 1}
      image_id: str
    """
    def __init__(self, df, img_dir='public/train', is_train=True, augment=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        self.augment = augment and is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.img_dir, f'{image_id}.jpg')
        
        # Load directly as single-channel grayscale (128, 800)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = img.astype(np.float32) / 255.0  # (128, 800)
        
        # Build 4-channel ground truth mask and 4-channel classification label
        masks = np.zeros((4, 128, 800), dtype=np.float32)
        cls_labels = np.zeros(4, dtype=np.float32)
        
        for i, d in enumerate(DEFECT_TYPES):
            rle = row.get(d, '')
            if pd.notna(rle) and str(rle).strip():
                masks[i] = rle_decode(rle)
                cls_labels[i] = 1.0
                
        # Data Augmentations
        if self.augment:
            # 1. Random Horizontal Flip (p=0.5)
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                masks = np.flip(masks, axis=2).copy()
                
            # 2. Random Vertical Flip (p=0.5)
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                masks = np.flip(masks, axis=1).copy()
                
            # 3. Random Contrast & Brightness Shift
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-0.10, 0.10)
                img = np.clip(alpha * img + beta, 0.0, 1.0)
                
            # 4. Coarse Dropout / Cutout (p=0.3)
            if np.random.rand() > 0.7:
                h, w = 128, 800
                ch, cw = np.random.randint(8, 24), np.random.randint(16, 48)
                y1 = np.random.randint(0, h - ch)
                x1 = np.random.randint(0, w - cw)
                img[y1:y1+ch, x1:x1+cw] = 0.0
                # Do not zero ground truth masks during cutout to teach occlusion robustness
                
        # Expand channel dimension for image: (1, 128, 800)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        cls_tensor = torch.from_numpy(cls_labels).float()
        mask_tensor = torch.from_numpy(masks).float()
        
        return {
            'image': img_tensor,
            'cls_label': cls_tensor,
            'mask': mask_tensor,
            'image_id': image_id
        }

class TestDefectDataset(Dataset):
    """
    Test Dataset for fast inference.
    """
    def __init__(self, image_ids, img_dir='public/test'):
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
