import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure src importable
if '.' not in sys.path:
    sys.path.append('.')

from src.dataset import prepare_image_dataframe, DefectDataset, DEFECT_TYPES
from src.models import DualHeadMTFPN
from src.losses import JointMultiTaskLoss
from src.trainer import Trainer
from src.metrics import compute_competition_metric
from src.postprocess import postprocess_mask

def train_production_model(
    epochs=10,
    batch_size=10,
    lr=5e-4,
    patience=3,
    fold=0,
    resume=True,
    device='cuda'
):
    print("=" * 75)
    print("🚀 PRODUCTION TRAINING: Dual-Head MT-FPN (ResNet-34 Grayscale Optimized)")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Resume: {resume}")
    print("=" * 75)
    
    # 1. Prepare 15-state stratified dataframe
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    
    # To maximize positive defect signal while controlling training time:
    # Use all positive defect images + balanced set of clean images
    train_fold_df = df[df['fold'] != fold].copy()
    val_fold_df = df[df['fold'] == fold].copy()
    
    # Filter train to include all positive images + 2,500 clean images (balanced curriculum)
    pos_train = train_fold_df[train_fold_df['strat_label'] != 'none']
    clean_train = train_fold_df[train_fold_df['strat_label'] == 'none'].sample(n=2500, random_state=42)
    balanced_train_df = pd.concat([pos_train, clean_train]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Validation evaluation set from Fold 0 (1,000 stratified samples for fast, robust per-epoch evaluation)
    val_eval_df = val_fold_df.sample(n=min(1000, len(val_fold_df)), random_state=42).reset_index(drop=True)
    
    print(f"Balanced Train Set: {len(balanced_train_df)} images ({len(pos_train)} defective + {len(clean_train)} clean)")
    print(f"Validation Evaluation Set: {len(val_eval_df)} images (Fold {fold} stratified)")
    
    train_ds = DefectDataset(balanced_train_df, img_dir='public/train', is_train=True, augment=True)
    val_ds = DefectDataset(val_eval_df, img_dir='public/train', is_train=False, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 2. Instantiate Model 3 (DualHeadMTFPN with 1-channel conv1)
    model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=True)
    
    # 3. Optimizer & Cosine Annealing Scheduler with Warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # 4. Multi-Task Criterion with positive class weighting to boost minority classes
    pos_weight = torch.tensor([2.0, 4.0, 1.0, 1.5], dtype=torch.float32)
    criterion = JointMultiTaskLoss(cls_weight=0.5, bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)
    
    ckpt_path = 'checkpoints/production/dualhead_mtfpn_resnet34_best.pth'
    resume_ckpt = ckpt_path if (resume and os.path.exists(ckpt_path)) else None
    thresh_path = 'checkpoints/production/best_thresholds.json'
    postprocess_cfg = thresh_path if os.path.exists(thresh_path) else None
    
    # 5. Trainer with streaming metric evaluation & memory hygiene
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir='checkpoints/production',
        experiment_name='dualhead_mtfpn_resnet34',
        patience=patience,
        min_delta=1e-4,
        resume_checkpoint=resume_ckpt,
        postprocess_cfg=postprocess_cfg
    )
    
    t0 = time.time()
    history = trainer.fit(epochs=epochs)
    total_time = time.time() - t0
    
    print(f"\n🎉 Production training complete in {total_time/60:.2f} minutes!")
    print(f"Best Validation Composite Score: {trainer.best_score:.5f} (Epoch {trainer.best_epoch})")
    
    return trainer, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    
    train_production_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        resume=args.resume
    )
