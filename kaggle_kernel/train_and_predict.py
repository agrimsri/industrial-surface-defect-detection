"""
Industrial Surface Defect Detection - Self-Contained Kaggle Training & Inference Pipeline
Model: Dual-Head Multi-Task FPN (ResNet-34 Grayscale Optimized)
Features:
- 1-channel Grayscale conv1 adaptation
- Multi-task auxiliary defect classification + FPN segmentation
- Positive class loss weighting (pos_weight) for minority defect classes (defect_a, defect_b)
- Cosine Annealing learning rate schedule
- Mixed precision AMP FP16 training with memory hygiene
- Out-of-fold class-specific threshold calibration (tau_cls, tau_prob, min_size)
- Test inference with Test-Time Augmentation (TTA)
- Exact quotation formatting (csv.QUOTE_ALL, ' ' for clean masks) matching Kaggle evaluator
"""

import os
import sys
import gc
import json
import time
import math
import argparse
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Ensure timm is installed
try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm"])
    import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. CONSTANTS & UTILITIES
# =============================================================================
DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']
DEFECT_TO_IDX = {d: i for i, d in enumerate(DEFECT_TYPES)}
IDX_TO_DEFECT = {i: d for i, d in enumerate(DEFECT_TYPES)}

DEFAULT_CLS_THRESHOLDS = {0: 0.50, 1: 0.35, 2: 0.50, 3: 0.45}
DEFAULT_PROB_THRESHOLDS = {0: 0.50, 1: 0.40, 2: 0.50, 3: 0.45}
DEFAULT_MIN_SIZES = {0: 50, 1: 95, 2: 40, 3: 100}

def rle_decode(rle_str, shape=(128, 800)):
    """Decodes Run-Length Encoded string to binary mask (Fortran order, 1-indexed)."""
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
    """Encodes binary mask (128, 800) to Run-Length Encoded string (Fortran order, 1-indexed)."""
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

def find_data_paths():
    """Locates train.csv, sample_submission.csv, train_dir, and test_dir dynamically and recursively."""
    print("🔍 Discovering dataset in filesystem...")
    
    # 1. Search recursively in /kaggle/input if available
    train_csv = None
    data_dir = None
    if os.path.exists('/kaggle/input'):
        print("  Scanning /kaggle/input contents:")
        for root, dirs, files in os.walk('/kaggle/input'):
            print(f"    - {root} (dirs: {len(dirs)}, files: {len(files)})")
            if 'train.csv' in files and train_csv is None:
                train_csv = os.path.join(root, 'train.csv')
                data_dir = root
                print(f"    ⭐ Found train.csv at: {train_csv}")
    
    # 2. Fallback to candidate local paths
    if train_csv is None:
        for cand in ['public', '.', '/kaggle/working']:
            p = os.path.join(cand, 'train.csv')
            if os.path.exists(p):
                train_csv = os.path.abspath(p)
                data_dir = os.path.dirname(train_csv)
                break
                
    if train_csv is None:
        fs_dump = []
        for p in ['/kaggle', '.']:
            if os.path.exists(p):
                for r, d, f in os.walk(p):
                    fs_dump.append(f"{r} ({len(f)} files)")
                    if len(fs_dump) > 30:
                        break
        raise FileNotFoundError(f"Could not locate train.csv! Scanned paths:\n" + "\n".join(fs_dump))
        
    # Locate sample_submission.csv
    sample_sub = os.path.join(data_dir, 'sample_submission.csv')
    if not os.path.exists(sample_sub):
        search_root = '/kaggle/input' if os.path.exists('/kaggle/input') else data_dir
        for root, _, files in os.walk(search_root):
            if 'sample_submission.csv' in files:
                sample_sub = os.path.join(root, 'sample_submission.csv')
                break
                
    # Locate train image directory
    train_dir = None
    for d in ['train', 'train_images']:
        p = os.path.join(data_dir, d)
        if os.path.isdir(p):
            train_dir = p
            break
    if train_dir is None:
        search_root = '/kaggle/input' if os.path.exists('/kaggle/input') else data_dir
        for root, dirs, _ in os.walk(search_root):
            for d in ['train', 'train_images']:
                if d in dirs:
                    train_dir = os.path.join(root, d)
                    break
            if train_dir:
                break
                
    # Locate test image directory
    test_dir = None
    for d in ['test', 'test_images']:
        p = os.path.join(data_dir, d)
        if os.path.isdir(p):
            test_dir = p
            break
    if test_dir is None:
        search_root = '/kaggle/input' if os.path.exists('/kaggle/input') else data_dir
        for root, dirs, _ in os.walk(search_root):
            for d in ['test', 'test_images']:
                if d in dirs:
                    test_dir = os.path.join(root, d)
                    break
            if test_dir:
                break
            
    print(f"📁 Located dataset paths:")
    print(f"   Data root:  {data_dir}")
    print(f"   Train CSV:  {train_csv}")
    print(f"   Sample Sub: {sample_sub}")
    print(f"   Train Dir:  {train_dir}")
    print(f"   Test Dir:   {test_dir}")
    
    return train_csv, sample_sub, train_dir, test_dir

# =============================================================================
# 2. DATASET & DATALOADERS
# =============================================================================
def prepare_image_dataframe(csv_path, n_splits=5, seed=42):
    """Pivots long-format train.csv into 1 row per unique image with 15-state stratification."""
    df = pd.read_csv(csv_path)
    pivoted = df.pivot(index='image_id', columns='defect_type', values='mask_rle').reset_index()
    pivoted.columns.name = None
    
    for d in DEFECT_TYPES:
        if d not in pivoted.columns:
            pivoted[d] = ''
        pivoted[f'{d}_has'] = pivoted[d].notna() & (pivoted[d].str.strip() != '')
        
    def get_strat_label(row):
        active = [str(i) for i, d in enumerate(DEFECT_TYPES) if row[f'{d}_has']]
        return '_'.join(active) if active else 'none'
        
    pivoted['strat_label'] = pivoted.apply(get_strat_label, axis=1)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pivoted['fold'] = -1
    for fold, (_, val_idx) in enumerate(skf.split(pivoted, pivoted['strat_label'])):
        pivoted.loc[val_idx, 'fold'] = fold
        
    return pivoted

class DefectDataset(Dataset):
    """Loads 1-channel grayscale image (1, 128, 800) with spatial & intensity augmentations."""
    def __init__(self, df, img_dir, is_train=True, augment=True):
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
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = img.astype(np.float32) / 255.0  # (128, 800)
        
        masks = np.zeros((4, 128, 800), dtype=np.float32)
        cls_labels = np.zeros(4, dtype=np.float32)
        
        for i, d in enumerate(DEFECT_TYPES):
            rle = row.get(d, '')
            if pd.notna(rle) and str(rle).strip():
                masks[i] = rle_decode(rle)
                cls_labels[i] = 1.0
                
        if self.augment:
            # Random Horizontal Flip (p=0.5)
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                masks = np.flip(masks, axis=2).copy()
                
            # Random Vertical Flip (p=0.5)
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                masks = np.flip(masks, axis=1).copy()
                
            # Random Brightness & Contrast jitter (p=0.5)
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-0.10, 0.10)
                img = np.clip(alpha * img + beta, 0.0, 1.0)
                
            # Coarse Dropout / Cutout (p=0.3)
            if np.random.rand() > 0.7:
                h, w = 128, 800
                ch, cw = np.random.randint(8, 24), np.random.randint(16, 48)
                y1 = np.random.randint(0, h - ch)
                x1 = np.random.randint(0, w - cw)
                img[y1:y1+ch, x1:x1+cw] = 0.0
                
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
    """Inference Dataset for fast test batching."""
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

# =============================================================================
# 3. MODEL ARCHITECTURE (Model 3: Dual-Head MT-FPN)
# =============================================================================
class DualHeadMTFPN(nn.Module):
    """
    Production-Grade Multi-Task Feature Pyramid Network.
    - 1-channel Grayscale Conv1 adaptation (66.7% memory & bandwidth cut)
    - Auxiliary Classification Head: Multi-label classification (p_k) from deepest layer C5
    - Feature Pyramid Network (FPN) Decoder: Multi-scale lateral 1x1 + top-down fusion
    - Output: (cls_logits [B, 4], seg_logits [B, 4, 128, 800])
    """
    def __init__(self, backbone_name='resnet34', num_classes=4, pretrained=True, fpn_dim=128):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # strides 4, 8, 16, 32
        )
        self.feature_channels = self.encoder.feature_info.channels()
        
        # Adapt first conv to 1-channel grayscale input
        self._adapt_first_conv_to_grayscale()
        
        # 1. Image-level Defect Classification Head (from C5)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(self.feature_channels[-1], 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
        # 2. FPN Lateral & Smoothing Layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_dim, kernel_size=1) for ch in self.feature_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.GELU()
            ) for _ in self.feature_channels
        ])
        
        # 3. Final Multi-Scale Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim * len(self.feature_channels), fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def _adapt_first_conv_to_grayscale(self):
        conv1 = None
        parent_name = ''
        attr_name = ''
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                conv1 = module
                if '.' in name:
                    parent_name, attr_name = name.rsplit('.', 1)
                else:
                    attr_name = name
                break
                
        if conv1 is not None:
            w_3ch = conv1.weight.data
            w_1ch = w_3ch.sum(dim=1, keepdim=True)
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=(conv1.bias is not None)
            )
            new_conv.weight.data.copy_(w_1ch)
            if conv1.bias is not None:
                new_conv.bias.data.copy_(conv1.bias.data)
                
            if parent_name:
                parent = dict(self.encoder.named_modules())[parent_name]
            else:
                parent = self.encoder
            setattr(parent, attr_name, new_conv)

    def forward(self, x):
        feats = self.encoder(x)  # C2, C3, C4, C5
        
        # Auxiliary Classification Prediction
        cls_logits = self.cls_head(feats[-1])  # (B, 4)
        
        # Top-down FPN pathway
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='nearest'
            )
            
        c2_shape = feats[0].shape[-2:]
        smoothed = [
            F.interpolate(sm(lat), size=c2_shape, mode='bilinear', align_corners=False)
            for sm, lat in zip(self.smooth_convs, laterals)
        ]
        
        fused = torch.cat(smoothed, dim=1)
        seg_feats = self.seg_head(fused)
        seg_logits = F.interpolate(seg_feats, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return cls_logits, seg_logits

# =============================================================================
# 4. LOSS FUNCTIONS
# =============================================================================
class SoftDiceLoss(nn.Module):
    """
    Differentiable Foreground Soft Dice Loss for binary segmentation.
    Only computes Dice loss over instances where foreground defect is present,
    matching Kaggle's foreground-only Dice evaluation specification.
    Prevents artificial 1.0 Dice penalties on empty slices from driving
    minority class (defect_a, defect_b) sigmoid probabilities to 0.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        t_sum = targets.sum(dim=2)
        p_sum = probs.sum(dim=2)
        intersection = (probs * targets).sum(dim=2)
        
        pos_mask = (t_sum > 0).float()
        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)
        
        num_pos = pos_mask.sum()
        if num_pos > 0:
            loss = (pos_mask * (1.0 - dice)).sum() / num_pos
        else:
            loss = torch.tensor(0.0, device=probs.device, requires_grad=True)
        return loss

class JointMultiTaskLoss(nn.Module):
    """
    Joint multi-task loss with positive class weighting (pos_weight)
    to boost recall on minority classes defect_a and defect_b.
    """
    def __init__(self, cls_weight=0.5, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.cls_weight = cls_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1))
        else:
            self.bce_cls = nn.BCEWithLogitsLoss()
            self.bce_seg = nn.BCEWithLogitsLoss()
            
        self.dice_seg = SoftDiceLoss()

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        l_cls = self.bce_cls(cls_logits, cls_targets)
        l_bce = self.bce_seg(seg_logits, seg_targets)
        probs = torch.sigmoid(seg_logits)
        l_dice = self.dice_seg(probs, seg_targets)
        
        l_seg = self.bce_weight * l_bce + self.dice_weight * l_dice
        l_total = l_seg + self.cls_weight * l_cls
        
        return {
            'total_loss': l_total,
            'seg_loss': l_seg,
            'cls_loss': l_cls,
            'bce_loss': l_bce,
            'dice_loss': l_dice
        }

# =============================================================================
# 5. POST-PROCESSING & METRIC ACCUMULATOR
# =============================================================================
def postprocess_mask(prob_mask,
                     cls_prob=None,
                     cls_idx=0,
                     cls_thresh=None,
                     prob_thresh=None,
                     min_size=None,
                     apply_morphology=True):
    """Two-Stage Gated Post-Processing Pipeline."""
    if isinstance(cls_thresh, dict):
        cls_thresh = cls_thresh.get(cls_idx, cls_thresh.get(str(cls_idx), DEFAULT_CLS_THRESHOLDS.get(cls_idx, 0.50)))
    elif cls_thresh is None:
        cls_thresh = DEFAULT_CLS_THRESHOLDS.get(cls_idx, 0.50)

    if isinstance(prob_thresh, dict):
        prob_thresh = prob_thresh.get(cls_idx, prob_thresh.get(str(cls_idx), DEFAULT_PROB_THRESHOLDS.get(cls_idx, 0.50)))
    elif prob_thresh is None:
        prob_thresh = DEFAULT_PROB_THRESHOLDS.get(cls_idx, 0.50)

    if isinstance(min_size, dict):
        min_size = min_size.get(cls_idx, min_size.get(str(cls_idx), DEFAULT_MIN_SIZES.get(cls_idx, 50)))
    elif min_size is None:
        min_size = DEFAULT_MIN_SIZES.get(cls_idx, 50)

    # Stage 1: Classification Gating
    if cls_prob is not None and cls_prob < cls_thresh:
        return np.zeros(prob_mask.shape, dtype=np.uint8)

    # Stage 2: Probability Thresholding
    binary = (prob_mask > prob_thresh).astype(np.uint8)
    if not np.any(binary):
        return binary

    # Stage 3: Class-Specific Morphological Refinement
    if apply_morphology:
        if cls_idx == 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif cls_idx == 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Stage 4: Connected Component Size Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean_mask[labels == lbl] = 1

    if clean_mask.sum() < min_size:
        clean_mask.fill(0)

    return clean_mask

class StreamingMetricMeter:
    """Streaming accumulator for Kaggle composite competition metric."""
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.pos_dices = []
        self.class_pos_dices = {c: [] for c in range(self.num_classes)}
        self.neg_fp_scores = []
        self.neg_empty_correct = []
        self.class_neg_fp = {c: [] for c in range(self.num_classes)}
        self.class_neg_empty = {c: [] for c in range(self.num_classes)}
        
    def update(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        B, C, H, W = targets.shape
        p_flat = (preds > 0).reshape(B, C, -1)
        t_flat = (targets > 0).reshape(B, C, -1)
        
        p_sums = p_flat.sum(axis=-1, dtype=np.int64)
        t_sums = t_flat.sum(axis=-1, dtype=np.int64)
        
        for b in range(B):
            for c in range(C):
                t_sum = int(t_sums[b, c])
                p_sum = int(p_sums[b, c])
                
                if t_sum > 0:
                    if p_sum == 0:
                        dice = 0.0
                    else:
                        inter = int(np.logical_and(p_flat[b, c], t_flat[b, c]).sum())
                        dice = float((2.0 * inter) / (p_sum + t_sum))
                    self.pos_dices.append(dice)
                    self.class_pos_dices[c].append(dice)
                else:
                    if p_sum == 0:
                        fp_score = 1.0
                        empty_score = 1.0
                    else:
                        fp_penalty = min(1.0, float(p_sum) / 500.0)
                        fp_score = max(0.0, 1.0 - fp_penalty)
                        empty_score = 0.0
                    self.neg_fp_scores.append(fp_score)
                    self.neg_empty_correct.append(empty_score)
                    self.class_neg_fp[c].append(fp_score)
                    self.class_neg_empty[c].append(empty_score)
                        
    def compute(self):
        s_dice = float(np.mean(self.pos_dices)) if len(self.pos_dices) > 0 else 0.0
        s_fp = float(np.mean(self.neg_fp_scores)) if len(self.neg_fp_scores) > 0 else 1.0
        s_empty = float(np.mean(self.neg_empty_correct)) if len(self.neg_empty_correct) > 0 else 1.0
        
        final_score = 0.70 * s_dice + 0.20 * s_fp + 0.10 * s_empty
        
        per_class_pos = {
            c: (float(np.mean(self.class_pos_dices[c])) if len(self.class_pos_dices[c]) > 0 else 0.0)
            for c in range(self.num_classes)
        }
        per_class_fp = {
            c: (float(np.mean(self.class_neg_fp[c])) if len(self.class_neg_fp[c]) > 0 else 1.0)
            for c in range(self.num_classes)
        }
        per_class_empty = {
            c: (float(np.mean(self.class_neg_empty[c])) if len(self.class_neg_empty[c]) > 0 else 1.0)
            for c in range(self.num_classes)
        }
        per_class_score = {
            c: 0.70 * per_class_pos[c] + 0.20 * per_class_fp[c] + 0.10 * per_class_empty[c]
            for c in range(self.num_classes)
        }
        
        return {
            'final_score': final_score,
            's_dice': s_dice,
            's_fp': s_fp,
            's_empty': s_empty,
            'per_class_dice': per_class_pos,
            'per_class_fp': per_class_fp,
            'per_class_empty': per_class_empty,
            'per_class_score': per_class_score,
            'n_pos': len(self.pos_dices),
            'n_neg': len(self.neg_fp_scores)
        }

# =============================================================================
# 6. TRAINING ENGINE
# =============================================================================
def train_model(
    train_csv_path,
    train_img_dir,
    epochs=7,
    batch_size=12,
    lr=5e-4,
    patience=4,
    device='cuda',
    save_dir='.'
):
    print("=" * 75)
    print("🚀 STARTING TRAINING: Dual-Head MT-FPN (ResNet-34 Grayscale)")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 75)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    df = prepare_image_dataframe(train_csv_path, n_splits=5, seed=42)
    
    # Balanced curriculum: all positive defect images + balanced set of clean images
    train_fold_df = df[df['fold'] != 0].copy()
    val_fold_df = df[df['fold'] == 0].copy()
    
    pos_train = train_fold_df[train_fold_df['strat_label'] != 'none']
    clean_train = train_fold_df[train_fold_df['strat_label'] == 'none'].sample(n=2500, random_state=42)
    balanced_train_df = pd.concat([pos_train, clean_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Validation evaluation set (1,000 stratified samples)
    val_eval_df = val_fold_df.sample(n=min(1000, len(val_fold_df)), random_state=42).reset_index(drop=True)
    
    print(f"Train set: {len(balanced_train_df)} images ({len(pos_train)} defective + {len(clean_train)} clean)")
    print(f"Val set:   {len(val_eval_df)} images")
    
    train_ds = DefectDataset(balanced_train_df, img_dir=train_img_dir, is_train=True, augment=True)
    val_ds = DefectDataset(val_eval_df, img_dir=train_img_dir, is_train=False, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if os.cpu_count() >= 4 else 2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Model 3
    model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=True).to(device)
    
    # Optimizer & Cosine Decay Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Loss: Pos-weighted BCE on defect_a & defect_b + Soft Dice
    pos_weight = torch.tensor([2.0, 4.0, 1.0, 1.5], dtype=torch.float32, device=device)
    criterion = JointMultiTaskLoss(cls_weight=0.5, bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight).to(device)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    best_score = -float('inf')
    best_epoch = -1
    best_ckpt_path = os.path.join(save_dir, 'dualhead_mtfpn_resnet34_best.pth')
    
    history = {'train_loss': [], 'val_loss': [], 'val_score': [], 'val_dice': []}
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True)
            cls_targets = batch['cls_label'].to(device, non_blocking=True)
            seg_targets = batch['mask'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                cls_logits, seg_logits = model(images)
                loss_dict = criterion(cls_logits, seg_logits, cls_targets, seg_targets)
                loss = loss_dict['total_loss']
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            del images, cls_targets, seg_targets, cls_logits, seg_logits, loss_dict, loss
            
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        meter = StreamingMetricMeter(num_classes=4)
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                cls_targets = batch['cls_label'].to(device, non_blocking=True)
                seg_targets = batch['mask'].to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                    cls_logits, seg_logits = model(images)
                    loss_dict = criterion(cls_logits, seg_logits, cls_targets, seg_targets)
                    val_loss += loss_dict['total_loss'].item()
                    
                cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
                seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
                targets_np = (seg_targets.cpu().numpy() > 0).astype(np.uint8)
                
                B, C, H, W = seg_probs.shape
                batch_preds = np.zeros((B, C, H, W), dtype=np.uint8)
                for b in range(B):
                    for c in range(C):
                        batch_preds[b, c] = postprocess_mask(
                            seg_probs[b, c],
                            cls_prob=cls_probs[b, c],
                            cls_idx=c
                        )
                meter.update(batch_preds, targets_np)
                del images, cls_targets, seg_targets, cls_logits, seg_logits, loss_dict, batch_preds
                
        metrics = meter.compute()
        val_score = metrics['final_score']
        val_loss /= len(val_loader)
        elapsed = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)
        history['val_dice'].append(metrics['s_dice'])
        
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Score: {val_score:.5f} (Dice: {metrics['s_dice']:.4f}, FP: {metrics['s_fp']:.4f}, Emp: {metrics['s_empty']:.4f})"
        )
        
        if val_score > best_score:
            improvement = val_score - best_score
            best_score = val_score
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': best_score,
                'val_metrics': metrics
            }, best_ckpt_path)
            print(f"  ⭐ Saved new best checkpoint (+{improvement:.5f}) to {best_ckpt_path}")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"\n🎉 Training complete! Best validation composite score: {best_score:.5f} (Epoch {best_epoch})")
    return best_ckpt_path, val_eval_df

# =============================================================================
# 7. OUT-OF-FOLD THRESHOLD CALIBRATION
# =============================================================================
def calibrate_thresholds(
    model_path,
    val_df,
    train_img_dir,
    device='cuda',
    save_path='best_thresholds.json'
):
    print("\n" + "=" * 75)
    print("🎯 OPTIMIZING TWO-STAGE POST-PROCESSING THRESHOLDS ON OUT-OF-FOLD DATA")
    print("=" * 75)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    val_ds = DefectDataset(val_df, img_dir=train_img_dir, is_train=False, augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    
    model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    all_cls_probs = []
    all_seg_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = (batch['mask'].numpy() > 0).astype(np.uint8)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
                cls_logits, seg_logits = model(images)
                
            cls_p = torch.sigmoid(cls_logits).cpu().numpy().astype(np.float32)
            seg_p = torch.sigmoid(seg_logits).cpu().numpy().astype(np.float16)
            
            all_cls_probs.append(cls_p)
            all_seg_probs.append(seg_p)
            all_targets.append(targets)
            del images, targets, cls_logits, seg_logits
            
    all_cls_probs = np.concatenate(all_cls_probs, axis=0)
    all_seg_probs = np.concatenate(all_seg_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    class_search_grids = {
        0: {  # defect_a: small localized pits
            'cls': [0.10, 0.20, 0.30, 0.40],
            'prob': [0.0001, 0.0003, 0.0005, 0.001, 0.005, 0.01],
            'min_size': [20, 30, 50, 75]
        },
        1: {  # defect_b: faint vertical scratches
            'cls': [0.05, 0.10, 0.15, 0.20],
            'prob': [0.00005, 0.0001, 0.0003, 0.0005, 0.001],
            'min_size': [15, 25, 40, 60]
        },
        2: {  # defect_c: large flake patches
            'cls': [0.30, 0.35, 0.40, 0.45, 0.50],
            'prob': [0.35, 0.40, 0.50, 0.55],
            'min_size': [40, 60, 75, 100]
        },
        3: {  # defect_d: rolling abrasion bands
            'cls': [0.20, 0.25, 0.30, 0.35, 0.40],
            'prob': [0.35, 0.40, 0.45, 0.50],
            'min_size': [50, 75, 100, 120]
        }
    }
    
    best_params = {}
    N = len(all_cls_probs)
    
    for c in range(4):
        c_name = DEFECT_TYPES[c]
        grid = class_search_grids[c]
        c_cls = all_cls_probs[:, c]
        c_seg = all_seg_probs[:, c]
        c_tgt = all_targets[:, c]
        
        t_sums = c_tgt.reshape(N, -1).sum(axis=1)
        pos_indices = np.where(t_sums > 0)[0]
        neg_indices = np.where(t_sums == 0)[0]
        
        best_c_score = -1.0
        best_c_config = None
        
        for ct in grid['cls']:
            for pt in grid['prob']:
                for ms in grid['min_size']:
                    pos_dices = []
                    neg_fp = []
                    neg_emp = []
                    
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
                                inter = np.logical_and(mask, c_tgt[idx]).sum()
                                pos_dices.append(float(2.0 * inter / (p_sum + t_sum)))
                                
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
                        best_c_config = {'cls_thresh': ct, 'prob_thresh': pt, 'min_size': ms, 'score': score}
                        
        best_params[c] = best_c_config
        print(f"  [{c_name:8s}] Best: tau_cls={best_c_config['cls_thresh']} | "
              f"tau_prob={best_c_config['prob_thresh']} | "
              f"min_size={best_c_config['min_size']}px -> Score: {best_c_score:.5f}")
        
    cfg_to_save = {
        'cls_thresh': {str(c): best_params[c]['cls_thresh'] for c in range(4)},
        'prob_thresh': {str(c): best_params[c]['prob_thresh'] for c in range(4)},
        'min_size': {str(c): best_params[c]['min_size'] for c in range(4)}
    }
    
    with open(save_path, 'w') as f:
        json.dump(cfg_to_save, f, indent=2)
    print(f"✅ Calibrated thresholds saved to {save_path}")
    return cfg_to_save

# =============================================================================
# 8. TEST INFERENCE & QUOTED SUBMISSION GENERATION
# =============================================================================
def generate_submission(
    model_path,
    sample_sub_path,
    test_img_dir,
    thresholds_cfg,
    output_path='submission.csv',
    use_tta=True,
    batch_size=12,
    device='cuda'
):
    print("\n" + "=" * 75)
    print("🚀 GENERATING TEST PREDICTIONS WITH TTA & CALIBRATED THRESHOLDS")
    print(f"Output: {output_path} | TTA: {use_tta} | Batch Size: {batch_size}")
    print("=" * 75)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    cls_thresh = thresholds_cfg.get('cls_thresh')
    prob_thresh = thresholds_cfg.get('prob_thresh')
    min_size = thresholds_cfg.get('min_size')
    
    model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    sub_template = pd.read_csv(sample_sub_path)
    test_image_ids = sub_template['image_id'].unique().tolist()
    print(f"Total test images: {len(test_image_ids)}")
    
    test_ds = TestDefectDataset(test_image_ids, img_dir=test_img_dir)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    rle_dict = {}
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
            
            B = len(img_ids)
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
                print(f"  Processed {processed_count}/{len(test_image_ids)} images ({pct:.1f}%) | {fps:.1f} FPS")
                
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Formatting submission rows matching sample_submission.csv...")
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
    print(f"✅ Saved submission to {output_path} (csv.QUOTE_ALL with ' ' for empty masks)")
    
    # Audit & Verification
    print("\n🔍 SUBMISSION AUDIT & INTEGRITY CHECK:")
    print(f"  Row count: {len(out_df)} (Target: 12000) -> {'PASS' if len(out_df) == 12000 else 'FAIL'}")
    print(f"  Columns: {list(out_df.columns)} -> {'PASS' if list(out_df.columns) == ['ID', 'image_id', 'defect_type', 'mask_rle'] else 'FAIL'}")
    print(f"  Null count: {out_df.isnull().sum().sum()} -> {'PASS' if out_df.isnull().sum().sum() == 0 else 'FAIL'}")
    
    empty_masks = (out_df['mask_rle'].str.strip() == '').sum()
    empty_pct = (empty_masks / len(out_df)) * 100
    pos_masks = len(out_df) - empty_masks
    print(f"  Empty Masks: {empty_masks:,} ({empty_pct:.2f}%)")
    print(f"  Positive Masks: {pos_masks:,} ({100 - empty_pct:.2f}%)")
    
    print("\n  Per-Class Defect Counts:")
    for d_type in DEFECT_TYPES:
        c_count = (out_df[out_df['defect_type'] == d_type]['mask_rle'].str.strip() != '').sum()
        print(f"    - {d_type:8s}: {c_count:5d} masks ({c_count/3000*100:.2f}% of test images)")
        assert c_count > 0, f"Error: Defect class {d_type} has 0 predicted instances!"
    print("  All 4 defect categories have active, non-zero predictions.")
    return output_path

# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=4)
    args, _ = parser.parse_known_args()
    
    train_csv, sample_sub, train_dir, test_dir = find_data_paths()
    
    # Set output destination
    work_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    
    # 1. Train
    best_ckpt, val_df = train_model(
        train_csv_path=train_csv,
        train_img_dir=train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        save_dir=work_dir
    )
    
    # 2. Calibrate thresholds on out-of-fold predictions
    thresh_json = os.path.join(work_dir, 'best_thresholds.json')
    calibrated_cfg = calibrate_thresholds(
        model_path=best_ckpt,
        val_df=val_df,
        train_img_dir=train_dir,
        save_path=thresh_json
    )
    
    # 3. Predict on test set & save submission.csv
    submission_path = os.path.join(work_dir, 'submission.csv')
    generate_submission(
        model_path=best_ckpt,
        sample_sub_path=sample_sub,
        test_img_dir=test_dir,
        thresholds_cfg=calibrated_cfg,
        output_path=submission_path,
        use_tta=True,
        batch_size=args.batch_size
    )
    print("\n🏁 FULL PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
