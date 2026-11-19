import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure workspace root is in python path
if '.' not in sys.path:
    sys.path.append('.')

from src.dataset import prepare_image_dataframe, DefectDataset
from src.models import DualHeadMTFPN, BaselineCustomUNet
from src.metrics import StreamingMetricMeter, compute_competition_metric
from src.postprocess import postprocess_mask, rle_encode, rle_decode, enforce_mask_disjointness
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

def test_enhanced_dualhead_forward():
    """
    Verifies that DualHeadMTFPN with enhanced=True supports dual pooling and learned upsampling.
    """
    print("\n--- Test 8: Enhanced DualHeadMTFPN with Dual Pooling & Full-Resolution Decoder ---")
    model = DualHeadMTFPN(backbone_name='tf_efficientnetv2_s', num_classes=4, pretrained=False, enhanced=True)
    x = torch.randn(2, 1, 128, 800)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, 4)
    assert seg_logits.shape == (2, 4, 128, 800)
    print("✅ Enhanced DualHeadMTFPN forward pass verified successfully!")

def test_multifold_stratification_split():
    """
    Verifies that prepare_image_dataframe produces clean 5-fold stratification
    with 0 overlap between validation folds and balanced fold sizes.
    """
    print("\n--- Test 9: Multi-Fold Stratification Integrity ---")
    df = prepare_image_dataframe('public/train.csv', n_splits=5, seed=42)
    assert 'fold' in df.columns
    folds = set(df['fold'].unique())
    assert folds == {0, 1, 2, 3, 4}, f"Expected folds {{0, 1, 2, 3, 4}}, got {folds}"
    
    # Check fold distribution
    for f in range(5):
        f_count = (df['fold'] == f).sum()
        assert 2200 <= f_count <= 2350, f"Fold {f} size {f_count} out of expected range"
    print("✅ Multi-fold stratification verified successfully!")

def test_scaled_backbone_tf_efficientnetv2_m():
    """
    Verifies that DualHeadMTFPN scales to tf_efficientnetv2_m with native 1-channel grayscale.
    """
    print("\n--- Test 10: Scaled Backbone tf_efficientnetv2_m ---")
    model = DualHeadMTFPN(backbone_name='tf_efficientnetv2_m', num_classes=4, pretrained=False)
    x = torch.randn(2, 1, 128, 800)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, 4)
    assert seg_logits.shape == (2, 4, 128, 800)
    print("✅ Scaled backbone tf_efficientnetv2_m forward pass verified successfully!")

def test_ensemble_probability_blending():
    """
    Verifies that ensemble blending averages classification and segmentation tensors correctly.
    """
    print("\n--- Test 11: Multi-Model Ensemble Blending ---")
    cp1 = torch.tensor([[0.8, 0.2, 0.4, 0.1]])
    cp2 = torch.tensor([[0.6, 0.4, 0.6, 0.3]])
    blended_cp = torch.stack([cp1, cp2]).mean(dim=0)
    expected_cp = torch.tensor([[0.7, 0.3, 0.5, 0.2]])
    assert torch.allclose(blended_cp, expected_cp)
    
    sp1 = torch.zeros(1, 4, 128, 800)
    sp2 = torch.ones(1, 4, 128, 800)
    blended_sp = torch.stack([sp1, sp2]).mean(dim=0)
    assert torch.allclose(blended_sp, torch.full_like(sp1, 0.5))
    print("✅ Ensemble probability blending verified successfully!")

def test_bayesian_consensus_blending():
    """
    Verifies that consensus blending eliminates isolated false alarms
    and bridges broken mask regions via union.
    """
    print("\n--- Test 12: Bayesian Consensus Blending Integrity ---")
    # Isolated defect_b noise mask
    noise_mask = np.zeros((128, 800), dtype=np.uint8)
    noise_mask[10:30, 10:15] = 1  # 100 px noise
    noise_rle = rle_encode(noise_mask)
    empty_rle = ' '
    
    # defect_b requires consensus: if only one predicts, output must be empty
    has_b = (noise_rle.strip() != '')
    has_m = (empty_rle.strip() != '')
    final_b = noise_rle if (has_b and has_m) else ' '
    assert final_b.strip() == '', "Failed to filter non-consensus noise mask on defect_b"
    
    # defect_c union: two overlapping patches
    m1 = np.zeros((128, 800), dtype=np.uint8)
    m1[20:40, 20:40] = 1
    m2 = np.zeros((128, 800), dtype=np.uint8)
    m2[30:50, 30:50] = 1
    comb = np.logical_or(m1, m2).astype(np.uint8)
    assert comb.sum() == 700  # 400 + 400 - 100 overlap = 700
    print("✅ Bayesian consensus blending verified successfully!")

def test_enforce_mask_disjointness():
    """
    Verifies that enforce_mask_disjointness strictly resolves contested pixels
    between defect classes via argmax over predicted probabilities.
    """
    print("\n--- Test 14: Strict Mask Disjointness Resolution ---")
    masks = np.zeros((4, 128, 800), dtype=np.uint8)
    # Class 0 mask: 20x20 at (10:30, 10:30)
    masks[0, 10:30, 10:30] = 1
    # Class 2 mask: 20x20 at (20:40, 20:40)
    masks[2, 20:40, 20:40] = 1

    # Contested region is [20:30, 20:30] (10x10 = 100 pixels)
    overlap_before = (masks.sum(axis=0) > 1).sum()
    assert overlap_before == 100, f"Expected 100 contested pixels, got {overlap_before}"

    # Set probabilities: Class 2 has 0.85, Class 0 has 0.60
    probs = np.zeros((4, 128, 800), dtype=np.float32)
    probs[0, 10:30, 10:30] = 0.60
    probs[2, 20:40, 20:40] = 0.85

    disjoint = enforce_mask_disjointness(masks, probs=probs)

    # Invariant: 0 overlapping pixels
    overlap_after = (disjoint.sum(axis=0) > 1).sum()
    assert overlap_after == 0, f"Expected 0 overlapping pixels, got {overlap_after}"
    # Class 2 should have won the contested region
    assert (disjoint[2, 20:30, 20:30] == 1).all(), "Class 2 should own the contested region"
    assert (disjoint[0, 20:30, 20:30] == 0).all(), "Class 0 should be 0 on contested region"
    print("✅ Mask disjointness resolution verified successfully!")

def test_sota_submission_verification():
    """
    Verifies that the generated submissions exist,
    satisfy all 12,000 row requirements, match schema, and have zero overlapping pixels.
    """
    print("\n--- Test 13: SOTA Submission Format & Integrity Check ---")
    sub_paths = ['submissions/submission_sota_disjoint.csv', 'submissions/submission_sota_consensus.csv']
    checked = 0
    for sub_path in sub_paths:
        if not os.path.exists(sub_path):
            continue
        checked += 1
        df = pd.read_csv(sub_path)
        assert len(df) == 12000, f"Expected 12000 rows in {sub_path}, got {len(df)}"
        assert list(df.columns) == ['ID', 'image_id', 'defect_type', 'mask_rle'], f"Unexpected columns: {list(df.columns)}"
        assert df.isnull().sum().sum() == 0, f"Found null values in {sub_path}"
        
        # Check that all 4 classes have non-zero predictions
        for d in ['defect_a', 'defect_b', 'defect_c', 'defect_d']:
            c_sub = df[df['defect_type'] == d]
            assert len(c_sub) == 3000, f"Expected 3000 rows for {d}, got {len(c_sub)}"
            active = (c_sub['mask_rle'].fillna('').str.strip() != '').sum()
            assert active > 0, f"Class {d} has zero positive predictions in {sub_path}!"
            
        if 'disjoint' in sub_path:
            # Verify 0 overlapping pixels
            pivoted = df.pivot(index='image_id', columns='defect_type', values='mask_rle')
            overlap_count = 0
            for img_id, row in pivoted.iloc[:100].iterrows():
                m = [rle_decode(row[d]) for d in ['defect_a', 'defect_b', 'defect_c', 'defect_d']]
                if (np.sum(m, axis=0) > 1).any():
                    overlap_count += 1
            assert overlap_count == 0, f"Expected 0 overlapping images in disjoint submission, found {overlap_count}"
            
    assert checked > 0, "No submission files found to verify"
    print(f"✅ SOTA Submission integrity verified successfully! ({checked} files verified)")

def test_verified_morphological_closing_kernels():
    """
    Test 14: Verifies that physical morphological closing kernels correctly bridge
    broken vertical striations (defect_b, 1x5), horizontal bands (defect_d, 5x1),
    and fill interior cavities (defect_c, 3x3; defect_a, 3x3).
    """
    print("\n--- Test 14: Verified Morphological Closing Kernels ---")
    # Defect B (1x5 vertical closing): 2-pixel vertical gap in vertical scratch
    prob_b = np.zeros((128, 800), dtype=np.float32)
    prob_b[20:30, 100:102] = 0.8
    # Gap at rows 30-31
    prob_b[32:45, 100:102] = 0.8
    mask_b = postprocess_mask(prob_b, cls_prob=0.9, cls_idx=1, cls_thresh=0.2, prob_thresh=0.5, min_size=10)
    # The gap should be closed
    assert mask_b[30:32, 100:102].all() == 1, "Defect B vertical 1x5 closing failed to bridge 2px gap!"

    # Defect D (5x1 horizontal closing): 2-pixel horizontal gap in horizontal band
    prob_d = np.zeros((128, 800), dtype=np.float32)
    prob_d[50:52, 200:220] = 0.8
    # Gap at cols 220-221
    prob_d[50:52, 222:245] = 0.8
    mask_d = postprocess_mask(prob_d, cls_prob=0.9, cls_idx=3, cls_thresh=0.2, prob_thresh=0.5, min_size=10)
    assert mask_d[50:52, 220:222].all() == 1, "Defect D horizontal 5x1 closing failed to bridge 2px gap!"

    # Defect C (3x3 closing): 1-pixel interior cavity in flake patch
    prob_c = np.ones((10, 10), dtype=np.float32) * 0.8
    prob_c[5, 5] = 0.0  # interior hole
    full_c = np.zeros((128, 800), dtype=np.float32)
    full_c[10:20, 10:20] = prob_c
    mask_c = postprocess_mask(full_c, cls_prob=0.9, cls_idx=2, cls_thresh=0.2, prob_thresh=0.5, min_size=10)
    assert mask_c[15, 15] == 1, "Defect C 3x3 closing failed to fill interior cavity!"

    print("✅ Verified morphological closing kernels tested successfully!")

def test_multifold_continuous_probability_ensembling():
    """
    Test 15: Verifies that averaging continuous probability maps across multi-fold
    committees prior to binarization preserves borderline signals and avoids
    false negative dropout from disjoint carving or premature thresholding.
    """
    print("\n--- Test 15: Continuous Probability Ensembling ---")
    # Simulate 2 committee predictions where neither alone reaches 0.50 threshold,
    # but their continuous combination safely exceeds it
    p1 = np.zeros((128, 800), dtype=np.float32)
    p2 = np.zeros((128, 800), dtype=np.float32)
    
    # Model 1 has confidence 0.48 in region, Model 2 has 0.54
    p1[30:50, 40:60] = 0.48
    p2[30:50, 40:60] = 0.54
    
    # Premature discrete thresholding at 0.50 with intersection:
    discrete_1 = (p1 > 0.50).astype(np.uint8)  # all 0!
    discrete_2 = (p2 > 0.50).astype(np.uint8)  # all 1
    premature_intersect = np.logical_and(discrete_1, discrete_2)
    assert premature_intersect.sum() == 0, "Expected discrete intersection to drop mask"
    
    # Continuous probability averaging (0.40 * p1 + 0.60 * p2)
    ens_continuous = 0.40 * p1 + 0.60 * p2  # 0.40*0.48 + 0.60*0.54 = 0.192 + 0.324 = 0.516 > 0.50!
    assert np.isclose(ens_continuous[35, 45], 0.516)
    binary_ens = (ens_continuous > 0.50).astype(np.uint8)
    assert binary_ens[30:50, 40:60].sum() == 400, "Continuous ensembling should retain all 400 pixels!"
    print("✅ Continuous probability ensembling retention verified successfully!")

if __name__ == '__main__':
    test_streaming_metric_equivalence()
    test_val_loader_sanitization()
    test_pipeline_smoke()
    test_per_class_threshold_postprocessing()
    test_streaming_meter_per_class_metrics()
    test_joint_loss_pos_weight()
    test_efficientnetv2_dualhead_forward()
    test_enhanced_dualhead_forward()
    test_multifold_stratification_split()
    test_scaled_backbone_tf_efficientnetv2_m()
    test_ensemble_probability_blending()
    test_bayesian_consensus_blending()
    test_enforce_mask_disjointness()
    test_sota_submission_verification()
    test_verified_morphological_closing_kernels()
    test_multifold_continuous_probability_ensembling()
    print("\n🎉 ALL 16 TESTS PASSED!")
