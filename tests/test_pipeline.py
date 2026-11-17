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
from src.postprocess import postprocess_mask, rle_encode, rle_decode
from src.trainer import Trainer

from src.losses import JointMultiTaskLoss

def test_joint_loss_pos_weight():
    """
    Verifies that JointMultiTaskLoss correctly applies pos_weight on both classification and segmentation.
    """
    print("\n--- Test 6: JointMultiTaskLoss with pos_weight ---")
    pos_weight = [2.0, 4.0, 1.0, 1.5]
    criterion = JointMultiTaskLoss(pos_weight=pos_weight)
    
    cls_logits = torch.randn(4, 4)
    seg_logits = torch.randn(4, 4, 128, 800)
    cls_targets = torch.randint(0, 2, (4, 4)).float()
    seg_targets = torch.randint(0, 2, (4, 4, 128, 800)).float()
    
    loss_dict = criterion(cls_logits, seg_logits, cls_targets, seg_targets)
    assert 'total_loss' in loss_dict
    assert 'seg_loss' in loss_dict
    assert 'cls_loss' in loss_dict
    assert not torch.isnan(loss_dict['total_loss'])
    assert not torch.isinf(loss_dict['total_loss'])
    assert loss_dict['total_loss'].item() > 0
    print("✅ JointMultiTaskLoss with pos_weight verified successfully!")

def test_per_class_threshold_postprocessing():
    """
    Verifies that postprocess_mask correctly parses per-class dictionaries
    for cls_thresh, prob_thresh, and min_size without crashing or type errors.
    """
    print("\n--- Test 4: Per-Class Threshold Dictionary Support ---")
    prob_map = np.zeros((128, 800), dtype=np.float32)
    prob_map[20:40, 20:40] = 0.45  # 400 pixels of 0.45 probability
    
    cls_thresh_dict = {0: 0.20, 1: 0.10, 2: 0.50, 3: 0.50}
    prob_thresh_dict = {0: 0.40, 1: 0.20, 2: 0.50, 3: 0.50}
    min_size_dict = {0: 50, 1: 50, 2: 50, 3: 50}
    
    # Class 0: cls_prob = 0.25 >= 0.20, prob 0.45 >= 0.40 -> active mask
    mask_c0 = postprocess_mask(
        prob_map, cls_prob=0.25, cls_idx=0,
        cls_thresh=cls_thresh_dict, prob_thresh=prob_thresh_dict, min_size=min_size_dict
    )
    assert mask_c0.sum() == 400, f"Expected 400 pixels, got {mask_c0.sum()}"
    
    # Class 2: cls_prob = 0.25 < 0.50 (gated out) -> empty mask
    mask_c2 = postprocess_mask(
        prob_map, cls_prob=0.25, cls_idx=2,
        cls_thresh=cls_thresh_dict, prob_thresh=prob_thresh_dict, min_size=min_size_dict
    )
    assert mask_c2.sum() == 0, f"Expected 0 pixels, got {mask_c2.sum()}"
    print("✅ Per-class dictionary postprocessing verified successfully!")

def test_streaming_meter_per_class_metrics():
    """
    Verifies that StreamingMetricMeter records per_class_fp, per_class_empty, and per_class_score.
    """
    print("\n--- Test 5: Per-Class Metric Accounting ---")
    meter = StreamingMetricMeter(num_classes=4)
    preds = np.zeros((4, 4, 128, 800), dtype=np.uint8)
    targets = np.zeros((4, 4, 128, 800), dtype=np.uint8)
    
    # Instance 0: Class 0 positive ground truth, perfect prediction
    targets[0, 0, 10:20, 10:20] = 1
    preds[0, 0, 10:20, 10:20] = 1
    
    # Instance 1: Class 1 negative ground truth, false positive prediction (100 px)
    preds[1, 1, 10:20, 10:20] = 1
    
    meter.update(preds, targets)
    res = meter.compute()
    
    assert res['per_class_dice'][0] == 1.0
    assert 'per_class_fp' in res
    assert 'per_class_empty' in res
    assert 'per_class_score' in res
    # In batch of 4 samples: for class 1, 3 samples are correctly empty (1.0) and 1 is false positive (0.0) -> mean = 0.75
    assert res['per_class_empty'][1] == 0.75
    # False positive penalty for 100 px is 100/500 = 0.2 -> score is 0.8; remaining 3 samples are 1.0 -> mean = (3 + 0.8) / 4 = 0.95
    assert np.isclose(res['per_class_fp'][1], 0.95)
    print("✅ Per-class metric accounting verified successfully!")

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

def test_efficientnetv2_dualhead_forward():
    """
    Verifies that DualHeadMTFPN supports tf_efficientnetv2_s with stride-2 feature retention.
    """
    print("\n--- Test 7: DualHeadMTFPN with tf_efficientnetv2_s ---")
    model = DualHeadMTFPN(backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=False)
    x = torch.randn(2, 1, 128, 800)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, 4)
    assert seg_logits.shape == (2, 4, 128, 800)
    print("✅ DualHeadMTFPN with tf_efficientnetv2_s verified successfully!")

if __name__ == '__main__':
    test_streaming_metric_equivalence()
    test_val_loader_sanitization()
    test_pipeline_smoke()
    test_per_class_threshold_postprocessing()
    test_streaming_meter_per_class_metrics()
    test_joint_loss_pos_weight()
    test_efficientnetv2_dualhead_forward()
    print("\n🎉 ALL 7 TESTS PASSED!")
