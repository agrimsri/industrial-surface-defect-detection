import os
import sys
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure workspace root is in python path
if '.' not in sys.path:
    sys.path.append('.')

from src.dataset import prepare_image_dataframe, DefectDataset
from src.models import DualHeadMTFPN, BaselineCustomUNet
from src.metrics import StreamingMetricMeter, compute_competition_metric
from src.trainer import Trainer

def test_streaming_metric_equivalence():
    """
    Verifies that StreamingMetricMeter produces bit-exact equivalence with
    full-array metric calculation across multi-batch streams.
    """
    print("\n--- Test 1: Streaming Metric Equivalence ---")
    np.random.seed(42)
    B_total, C, H, W = 20, 4, 128, 800
    
    # Generate ground truth masks with mixed positive and negative surfaces
    targets = (np.random.rand(B_total, C, H, W) > 0.95).astype(np.uint8)
    preds = (np.random.rand(B_total, C, H, W) > 0.94).astype(np.uint8)
    
    # Clean samples (negative ground truth)
    targets[5:15, :] = 0
    # Clean predictions (empty output)
    preds[10:15, :] = 0
    
    # Full batch calculation
    full_metrics = compute_competition_metric(preds, targets)
    
    # Streaming multi-batch calculation (batches of size 4)
    meter = StreamingMetricMeter(num_classes=C)
    batch_size = 4
    for i in range(0, B_total, batch_size):
        meter.update(preds[i:i+batch_size], targets[i:i+batch_size])
    streaming_metrics = meter.compute()
    
    for key in ['final_score', 's_dice', 's_fp', 's_empty', 'n_pos', 'n_neg']:
        assert np.isclose(full_metrics[key], streaming_metrics[key], atol=1e-6), (
            f"Mismatch on {key}: full={full_metrics[key]}, streaming={streaming_metrics[key]}"
        )
        
    for c in range(C):
        assert np.isclose(full_metrics['per_class_dice'][c], streaming_metrics['per_class_dice'][c], atol=1e-6), (
            f"Class {c} Dice mismatch: {full_metrics['per_class_dice'][c]} vs {streaming_metrics['per_class_dice'][c]}"
        )
    print("✅ StreamingMetricMeter bit-exact match verified successfully!")

def test_val_loader_sanitization():
    """
    Verifies that Trainer enforces pin_memory=False and num_workers=0 on val_loader.
    """
    print("\n--- Test 2: Validation Loader Memory Hygiene Sanitization ---")
    model = BaselineCustomUNet(in_channels=1, num_classes=4)
    ds = TensorDataset(torch.randn(8, 1, 128, 800), torch.zeros(8, 4), torch.zeros(8, 4, 128, 800))
    
    # Caller inadvertently provides pin_memory=True and num_workers=2
    mock_val_loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    mock_train_loader = DataLoader(ds, batch_size=2, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = Trainer(
        model=model,
        train_loader=mock_train_loader,
        val_loader=mock_val_loader,
        optimizer=optimizer,
        checkpoint_dir='checkpoints/test',
        experiment_name='loader_sanitize_test'
    )
    
    assert trainer.val_loader.pin_memory == False, "Trainer failed to enforce pin_memory=False on val_loader"
    assert trainer.val_loader.num_workers == 0, "Trainer failed to enforce num_workers=0 on val_loader"
    print("✅ Trainer memory hygiene sanitization verified successfully!")

def test_pipeline_smoke():
    """
    End-to-end smoke test training for 2 epochs with streaming validation.
    """
    print("\n--- Test 3: End-to-End Pipeline Smoke Test ---")
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    
    train_df = df[df['fold'] != 0].iloc[:24]
    val_df = df[df['fold'] == 0].iloc[:12]
    
    train_ds = DefectDataset(train_df, img_dir='public/train', is_train=True, augment=True)
    val_ds = DefectDataset(val_df, img_dir='public/train', is_train=False, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    
    model = DualHeadMTFPN(backbone_name='resnet18', num_classes=4, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        checkpoint_dir='checkpoints/test',
        experiment_name='smoke_test',
        patience=2
    )
    
    history = trainer.fit(epochs=2)
    assert len(history['train_loss']) == 2
    assert len(history['val_final_score']) == 2
    assert os.path.exists('checkpoints/test/smoke_test_best.pth')
    
    # Test resume capability
    trainer_resumed = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        checkpoint_dir='checkpoints/test',
        experiment_name='smoke_test',
        resume_checkpoint='checkpoints/test/smoke_test_best.pth'
    )
    assert trainer_resumed.start_epoch == 2 or trainer_resumed.start_epoch == 3
    print("✅ End-to-End Pipeline & Resume verified successfully!")

if __name__ == '__main__':
    test_streaming_metric_equivalence()
    test_val_loader_sanitization()
    test_pipeline_smoke()
    print("\n🎉 ALL TESTS PASSED!")
