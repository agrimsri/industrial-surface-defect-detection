import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import prepare_image_dataframe, DefectDataset, DEFECT_TYPES
from src.models import BaselineCustomUNet, StandardTransferUNet, DualHeadMTFPN
from src.losses import JointMultiTaskLoss
from src.trainer import Trainer
from src.metrics import compute_competition_metric
from src.postprocess import postprocess_mask

def run_comparative_experiment(
    sample_limit=800,
    val_limit=400,
    epochs=3,
    batch_size=8,
    device='cuda'
):
    """
    Runs rapid fail-fast comparative experiments across the 3 core model architectures:
    - Model 1: Baseline Custom Scratch U-Net
    - Model 2: Standard Transfer Learning U-Net (ResNet18)
    - Model 3: Dual-Head Multi-Task FPN (ResNet34 with grayscale conv1 adaptation)
    """
    print("=" * 70)
    print("🚀 RUNNING THREE-TIER ARCHITECTURAL BENCHMARK EXPERIMENT")
    print(f"Sample limit: {sample_limit} train / {val_limit} val | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 70)
    
    # Load dataset with 15-state stratification
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    train_df = df[df['fold'] != 0].iloc[:sample_limit]
    val_df = df[df['fold'] == 0].iloc[:val_limit]
    
    train_ds = DefectDataset(train_df, is_train=True, augment=True)
    val_ds = DefectDataset(val_df, is_train=False, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    models_to_test = [
        ("Model 1: Custom Scratch U-Net", BaselineCustomUNet(in_channels=1, num_classes=4)),
        ("Model 2: Transfer Learning (ResNet18)", StandardTransferUNet(backbone_name='resnet18', num_classes=4, pretrained=True)),
        ("Model 3: Dual-Head MT-FPN (ResNet34)", DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=True))
    ]
    
    results = {}
    
    for name, model in models_to_test:
        print(f"\n---> Benchmarking {name}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir='checkpoints/experiments',
            experiment_name=name.split(':')[0].replace(' ', '_').lower(),
            patience=3
        )
        
        t0 = time.time()
        history = trainer.fit(epochs=epochs)
        elapsed = time.time() - t0
        
        results[name] = {
            'best_val_score': trainer.best_score,
            'best_epoch': trainer.best_epoch,
            'val_dice': history['val_dice'][-1] if history['val_dice'] else 0.0,
            'val_fp_score': history['val_fp_score'][-1] if history['val_fp_score'] else 0.0,
            'val_empty_acc': history['val_empty_acc'][-1] if history['val_empty_acc'] else 0.0,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
            'grad_norm': history['grad_norm'][-1] if history['grad_norm'] else 0.0,
            'elapsed_sec': elapsed
        }
        
    print("\n" + "=" * 70)
    print("📊 ARCHITECTURAL BENCHMARK RESULTS SUMMARY:")
    print("=" * 70)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    
    os.makedirs('logs', exist_ok=True)
    with open('logs/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == '__main__':
    run_comparative_experiment(sample_limit=800, val_limit=400, epochs=3, batch_size=8)
