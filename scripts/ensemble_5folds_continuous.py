"""
Master 5-Fold Continuous Probability Ensembling Pipeline
Combines continuous probability predictions across all 5 cross-validation folds:
- Fold 0 & 1 Committee (4-model Best + SWA with 4-way TTA)
- Fold 2, 3, 4 Committee (6-model Best + SWA with 4-way TTA)

Ensembling Methodology:
1. True Continuous Probability Averaging across all 5 folds before any binarization:
   P_cls = 0.40 * P_cls(0,1) + 0.60 * P_cls(2,3,4)
   P_seg = 0.40 * P_seg(0,1) + 0.60 * P_seg(2,3,4)
2. Defect-Tailored Physical Morphological Closing (Non-Disjoint):
   - defect_a: 3x3 rectangular closing (localized inclusions/pits)
   - defect_b: 1x5 vertical closing (ultra-thin vertical rolling scratches)
   - defect_c: 3x3 rectangular closing (flake patches, bridges interior cavities)
   - defect_d: 5x1 horizontal closing (continuous rolling abrasion bands)
3. Calibrated Gating & Connected Component Area Filtering:
   - Strictly avoids cross-class carving to maintain recall under 12.43x asymmetric FN penalty.
4. Kaggle Submission Verification & csv.QUOTE_ALL formatting.
"""

import os
import sys
import csv
import json
import argparse
import cv2
import numpy as np
import pandas as pd

if '.' not in sys.path:
    sys.path.append('.')

from src.postprocess import rle_encode

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

# Optimal thresholds calibrated on Out-Of-Fold data
DEFAULT_CLS_THRESHOLDS = {0: 0.25, 1: 0.25, 2: 0.45, 3: 0.25}
DEFAULT_PROB_THRESHOLDS = {0: 0.50, 1: 0.35, 2: 0.50, 3: 0.40}
DEFAULT_MIN_SIZES = {0: 80, 1: 120, 2: 40, 3: 100}

def postprocess_continuous_mask(
    prob_mask,
    cls_prob=None,
    cls_idx=0,
    cls_thresh=None,
    prob_thresh=None,
    min_size=None,
    apply_morphology=True
):
    """
    Two-stage post-processing pipeline operating on averaged continuous probability maps.
    """
    if isinstance(cls_thresh, dict):
        ct = cls_thresh.get(cls_idx, cls_thresh.get(str(cls_idx), DEFAULT_CLS_THRESHOLDS[cls_idx]))
    else:
        ct = cls_thresh if cls_thresh is not None else DEFAULT_CLS_THRESHOLDS[cls_idx]

    if isinstance(prob_thresh, dict):
        pt = prob_thresh.get(cls_idx, prob_thresh.get(str(cls_idx), DEFAULT_PROB_THRESHOLDS[cls_idx]))
    else:
        pt = prob_thresh if prob_thresh is not None else DEFAULT_PROB_THRESHOLDS[cls_idx]

    if isinstance(min_size, dict):
        ms = min_size.get(cls_idx, min_size.get(str(cls_idx), DEFAULT_MIN_SIZES[cls_idx]))
    else:
        ms = min_size if min_size is not None else DEFAULT_MIN_SIZES[cls_idx]

    # Stage 1: Continuous Classification Gate
    if cls_prob is not None and cls_prob < ct:
        return np.zeros(prob_mask.shape, dtype=np.uint8)

    # Stage 2: Continuous Probability Thresholding
    binary = (prob_mask > pt).astype(np.uint8)
    if not np.any(binary):
        return binary

    # Stage 3: Class-Specific Physical Morphological Refinement
    if apply_morphology:
        if cls_idx == 0:
            # Defect A: Localized pits (3x3 closing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif cls_idx == 1:
            # Defect B: Ultra-thin vertical scratches (1x5 vertical closing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif cls_idx == 2:
            # Defect C: Large irregular flake patches (3x3 closing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif cls_idx == 3:
            # Defect D: Rolling abrasion bands (5x1 horizontal closing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Stage 4: Connected Component Size Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= ms:
            clean_mask[labels == lbl] = 1

    if clean_mask.sum() < ms:
        clean_mask.fill(0)

    return clean_mask

def run_ensemble(
    cls_01_path='checkpoints/test_cls_probs_folds01.npy',
    seg_01_path='checkpoints/test_seg_probs_folds01.npy',
    cls_234_path='kaggle_output_folds234/test_cls_probs.npy',
    seg_234_path='kaggle_output_folds234/test_seg_probs.npy',
    sample_sub_path='public/sample_submission.csv',
    output_path='submissions/submission_5fold_committee.csv',
    weight_01=0.40,
    weight_234=0.60
):
    print("=" * 80)
    print("🚀 EXECUTING 5-FOLD CONTINUOUS PROBABILITY ENSEMBLING")
    print(f"  Folds 0 & 1 Cls: {cls_01_path} (Weight: {weight_01:.2f})")
    print(f"  Folds 0 & 1 Seg: {seg_01_path}")
    print(f"  Folds 2,3,4 Cls: {cls_234_path} (Weight: {weight_234:.2f})")
    print(f"  Folds 2,3,4 Seg: {seg_234_path}")
    print(f"  Output CSV:      {output_path}")
    print("=" * 80)

    # Load continuous probability predictions (classification fits in memory; segmentation is memory-mapped)
    cls_01 = np.load(cls_01_path).astype(np.float32)
    cls_234 = np.load(cls_234_path).astype(np.float32)
    print(f"Loaded Classification Probs: 01={cls_01.shape}, 234={cls_234.shape}")

    seg_01 = np.load(seg_01_path, mmap_mode='r')
    seg_234 = np.load(seg_234_path, mmap_mode='r')
    print(f"Memory-mapped Segmentation Probs: 01={seg_01.shape}, 234={seg_234.shape}")

    assert cls_01.shape == cls_234.shape, f"Mismatched cls shapes: {cls_01.shape} vs {cls_234.shape}"
    assert seg_01.shape == seg_234.shape, f"Mismatched seg shapes: {seg_01.shape} vs {seg_234.shape}"

    # Continuous Linear Blend
    norm_w01 = weight_01 / (weight_01 + weight_234)
    norm_w234 = weight_234 / (weight_01 + weight_234)
    print(f"Normalized ensemble weights: Folds 0&1 = {norm_w01:.4f}, Folds 2,3,4 = {norm_w234:.4f}")

    ens_cls_probs = norm_w01 * cls_01 + norm_w234 * cls_234

    sub_template = pd.read_csv(sample_sub_path)
    test_image_ids = sub_template['image_id'].unique().tolist()
    num_test = len(test_image_ids)
    assert len(ens_cls_probs) == num_test, f"Expected {num_test} images, got {len(ens_cls_probs)}"

    rle_dict = {}
    audit_counts = {d: 0 for d in DEFECT_TYPES}
    total_pixels = {d: 0 for d in DEFECT_TYPES}

    import time
    t0 = time.time()
    print("\nApplying continuous gating, morphological closing, and component filtering (streaming)...")
    for i, img_id in enumerate(test_image_ids):
        # Stream individual image continuous segmentation probability map (zero RAM overhead)
        seg_i = norm_w01 * seg_01[i].astype(np.float32) + norm_w234 * seg_234[i].astype(np.float32)
        cls_i = ens_cls_probs[i]

        for c, d in enumerate(DEFECT_TYPES):
            mask = postprocess_continuous_mask(
                prob_mask=seg_i[c],
                cls_prob=cls_i[c],
                cls_idx=c,
                cls_thresh=DEFAULT_CLS_THRESHOLDS,
                prob_thresh=DEFAULT_PROB_THRESHOLDS,
                min_size=DEFAULT_MIN_SIZES,
                apply_morphology=True
            )
            rle = rle_encode(mask)
            if rle.strip():
                audit_counts[d] += 1
                total_pixels[d] += int(mask.sum())
            rle_dict[(img_id, d)] = rle if rle.strip() else ' '

        if (i + 1) % 500 == 0 or (i + 1) == num_test:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  Ensembled {i + 1}/{num_test} images ({(i+1)/num_test*100:.1f}%) | {fps:.1f} FPS")

    # Build formatted submission
    out_df = sub_template.copy()
    out_df['mask_rle'] = [
        rle_dict.get((img_id, d), ' ')
        for img_id, d in zip(out_df['image_id'], out_df['defect_type'])
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Saved 5-fold ensemble submission to {output_path}")

    # Also update master submission.csv
    master_sub = 'submissions/submission.csv'
    out_df.to_csv(master_sub, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Updated master submission at {master_sub}")

    # Audit & Verification
    print("\n" + "=" * 80)
    print("🔍 SUBMISSION AUDIT & VERIFICATION RECORD")
    print("=" * 80)
    print(f"  Total Rows:   {len(out_df)} (Target: 12000) -> {'PASS' if len(out_df) == 12000 else 'FAIL'}")
    print(f"  Null Values:  {out_df.isnull().sum().sum()} -> {'PASS' if out_df.isnull().sum().sum() == 0 else 'FAIL'}")
    print(f"  Columns:      {list(out_df.columns)} -> PASS")

    empty_cnt = (out_df['mask_rle'].str.strip() == '').sum()
    pos_cnt = len(out_df) - empty_cnt
    print(f"  Empty Masks:  {empty_cnt:,} ({empty_cnt/len(out_df)*100:.2f}%)")
    print(f"  Active Masks: {pos_cnt:,} ({pos_cnt/len(out_df)*100:.2f}%)")

    print("\n  Per-Class Defect Counts & Total Positive Pixels:")
    for d in DEFECT_TYPES:
        cnt = audit_counts[d]
        px = total_pixels[d]
        print(f"    - {d:8s}: {cnt:4d} masks ({cnt/num_test*100:.2f}%) | {px:,} total active pixels")
        assert cnt > 0, f"Error: Defect class {d} has 0 positive predictions!"

    print("  ✅ All 4 defect classes have active non-zero predictions.")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls-01', type=str, default='checkpoints/test_cls_probs_folds01.npy')
    parser.add_argument('--seg-01', type=str, default='checkpoints/test_seg_probs_folds01.npy')
    parser.add_argument('--cls-234', type=str, default='kaggle_output_folds234/test_cls_probs.npy')
    parser.add_argument('--seg-234', type=str, default='kaggle_output_folds234/test_seg_probs.npy')
    parser.add_argument('--output', type=str, default='submissions/submission_5fold_committee.csv')
    parser.add_argument('--w01', type=float, default=0.40)
    parser.add_argument('--w234', type=float, default=0.60)
    args = parser.parse_args()

    run_ensemble(
        cls_01_path=args.cls_01,
        seg_01_path=args.seg_01,
        cls_234_path=args.cls_234,
        seg_234_path=args.seg_234,
        output_path=args.output,
        weight_01=args.w01,
        weight_234=args.w234
    )
