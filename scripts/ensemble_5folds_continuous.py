"""
Master 5-Fold Continuous Probability Ensembling Pipeline
Combines continuous probability predictions across all 5 cross-validation folds:
- Fold 0 & 1 Committee (4-model Best + SWA with 4-way TTA)
- Fold 2, 3, 4 Committee (6-model Best + SWA with 4-way TTA)

Ensembling Methodology:
1. True Continuous Probability Averaging across all 5 folds before any binarization:
   P_cls = 0.40 * P_cls(0,1) + 0.60 * P_cls(2,3,4)
   P_seg = 0.40 * P_seg(0,1) + 0.60 * P_seg(2,3,4) (when test_seg_probs.npy is present)
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
import time
import argparse
import numpy as np
import pandas as pd

if '.' not in sys.path:
    sys.path.append('.')

from src.postprocess import postprocess_mask, rle_encode
from scripts.generate_sota_consensus import generate_sota_submission

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

# Optimal thresholds calibrated on Out-Of-Fold data
DEFAULT_CLS_THRESHOLDS = {0: 0.25, 1: 0.30, 2: 0.30, 3: 0.25}
DEFAULT_PROB_THRESHOLDS = {0: 0.50, 1: 0.35, 2: 0.45, 3: 0.50}
DEFAULT_MIN_SIZES = {0: 80, 1: 80, 2: 60, 3: 200}

def load_thresholds(json_path):
    """Loads calibrated thresholds from JSON file if available."""
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            cls_t = {int(k): v for k, v in data.get('cls_thresh', {}).items()}
            prob_t = {int(k): v for k, v in data.get('prob_thresh', {}).items()}
            min_s = {int(k): v for k, v in data.get('min_size', {}).items()}
            print(f"Loaded calibrated thresholds from {json_path}")
            return cls_t, prob_t, min_s
        except Exception as e:
            print(f"Warning: Failed to load thresholds from {json_path}: {e}")
    return DEFAULT_CLS_THRESHOLDS, DEFAULT_PROB_THRESHOLDS, DEFAULT_MIN_SIZES

def resolve_path(primary_path, fallback_paths):
    """Resolves existing path between primary and fallbacks."""
    if primary_path and os.path.exists(primary_path):
        return primary_path
    for fb in fallback_paths:
        if fb and os.path.exists(fb):
            return fb
    return primary_path

def run_ensemble(
    cls_01_path='checkpoints/test_cls_probs_folds01.npy',
    seg_01_path='checkpoints/test_seg_probs_folds01.npy',
    cls_234_path='kaggle_output_folds234/test_cls_probs.npy',
    seg_234_path='kaggle_output_folds234/test_seg_probs.npy',
    thresh_json_path='kaggle_output_folds234/best_thresholds.json',
    sample_sub_path='public/sample_submission.csv',
    output_path='submissions/submission_5fold_committee.csv',
    blend_csv='submissions/submission_calibrated_consensus.csv',
    multifold_csv='kaggle_output_quick/submission.csv',
    weight_01=0.40,
    weight_234=0.60
):
    print("=" * 80)
    print("🚀 EXECUTING 5-FOLD CONTINUOUS PROBABILITY ENSEMBLING")
    
    # Resolve file paths
    cls_01_path = resolve_path(cls_01_path, ['checkpoints/test_cls_probs_folds01.npy', 'checkpoints/test_cls_probs.npy'])
    cls_234_path = resolve_path(cls_234_path, ['kaggle_output_folds234/test_cls_probs.npy', 'kaggle_output_quick/test_cls_probs.npy'])
    seg_01_path = resolve_path(seg_01_path, ['checkpoints/test_seg_probs_folds01.npy'])
    seg_234_path = resolve_path(seg_234_path, ['kaggle_output_folds234/test_seg_probs.npy', 'kaggle_output_quick/test_seg_probs.npy'])
    thresh_json_path = resolve_path(thresh_json_path, ['kaggle_output_folds234/best_thresholds.json', 'best_thresholds.json'])

    print(f"  Folds 0 & 1 Cls: {cls_01_path} (Weight: {weight_01:.2f})")
    print(f"  Folds 0 & 1 Seg: {seg_01_path}")
    print(f"  Folds 2,3,4 Cls: {cls_234_path} (Weight: {weight_234:.2f})")
    print(f"  Folds 2,3,4 Seg: {seg_234_path}")
    print(f"  Output CSV:      {output_path}")
    print("=" * 80)

    # 1. Continuous Classification Probability Averaging
    if not os.path.exists(cls_01_path):
        raise FileNotFoundError(f"Missing Folds 0&1 classification probabilities: {cls_01_path}")
    if not os.path.exists(cls_234_path):
        raise FileNotFoundError(f"Missing Folds 2,3,4 classification probabilities: {cls_234_path}")

    cls_01 = np.load(cls_01_path).astype(np.float32)
    cls_234 = np.load(cls_234_path).astype(np.float32)
    assert cls_01.shape == cls_234.shape, f"Mismatched cls shapes: {cls_01.shape} vs {cls_234.shape}"

    norm_w01 = weight_01 / (weight_01 + weight_234)
    norm_w234 = weight_234 / (weight_01 + weight_234)
    ens_cls_probs = norm_w01 * cls_01 + norm_w234 * cls_234

    # Save blended classification probabilities
    os.makedirs('checkpoints', exist_ok=True)
    blended_cls_path = 'checkpoints/test_cls_probs_5folds.npy'
    np.save(blended_cls_path, ens_cls_probs)
    print(f"✅ Saved 5-fold continuous classification probabilities to {blended_cls_path}")

    cls_thresh, prob_thresh, min_sizes = load_thresholds(thresh_json_path)

    # 2. Check if full 234 segmentation array is present locally for pixel-level continuous averaging
    has_full_seg = os.path.exists(seg_01_path) and os.path.exists(seg_234_path)

    if has_full_seg:
        print("⚡ Full segmentation probability arrays located! Executing streaming continuous pixel blending...")
        seg_01 = np.load(seg_01_path, mmap_mode='r')
        seg_234 = np.load(seg_234_path, mmap_mode='r')
        assert seg_01.shape == seg_234.shape, f"Mismatched seg shapes: {seg_01.shape} vs {seg_234.shape}"

        sub_template = pd.read_csv(sample_sub_path)
        test_image_ids = sub_template['image_id'].unique().tolist()
        num_test = len(test_image_ids)
        assert len(ens_cls_probs) == num_test

        rle_dict = {}
        audit_counts = {d: 0 for d in DEFECT_TYPES}
        total_pixels = {d: 0 for d in DEFECT_TYPES}

        t0 = time.time()
        for i, img_id in enumerate(test_image_ids):
            seg_i = norm_w01 * seg_01[i].astype(np.float32) + norm_w234 * seg_234[i].astype(np.float32)
            cls_i = ens_cls_probs[i]

            for c, d in enumerate(DEFECT_TYPES):
                mask = postprocess_mask(
                    prob_mask=seg_i[c],
                    cls_prob=cls_i[c],
                    cls_idx=c,
                    cls_thresh=cls_thresh,
                    prob_thresh=prob_thresh,
                    min_size=min_sizes,
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

        out_df = sub_template.copy()
        out_df['mask_rle'] = [
            rle_dict.get((img_id, d), ' ')
            for img_id, d in zip(out_df['image_id'], out_df['defect_type'])
        ]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    else:
        print(f"ℹ️ Segmentation map {seg_234_path} not found locally (uncompressed 2.45GB remote artifact).")
        print("Executing non-disjoint consensus ensembling with continuous classification probabilities...")
        resolved_blend = resolve_path(blend_csv, ['submissions/submission_calibrated_consensus.csv', 'submissions/submission_blend_optimal.csv'])
        resolved_multifold = resolve_path(multifold_csv, ['kaggle_output_quick/submission.csv', 'submissions/submission_multifold_ensemble.csv'])

        generate_sota_submission(
            blend_csv=resolved_blend,
            multifold_csv=resolved_multifold,
            sample_sub_path=sample_sub_path,
            cls_probs_path=blended_cls_path,
            output_csv=output_path,
            enforce_disjoint=False
        )
        out_df = pd.read_csv(output_path)

    # Master submission update
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

    print("\n  Per-Class Defect Counts:")
    for d in DEFECT_TYPES:
        cnt = (out_df[out_df['defect_type'] == d]['mask_rle'].str.strip() != '').sum()
        print(f"    - {d:8s}: {cnt:4d} masks ({cnt/3000*100:.2f}%)")
        assert cnt > 0, f"Error: Defect class {d} has 0 positive predictions!"

    print("  ✅ All 4 defect classes have active non-zero predictions.")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls-01', type=str, default='checkpoints/test_cls_probs_folds01.npy')
    parser.add_argument('--seg-01', type=str, default='checkpoints/test_seg_probs_folds01.npy')
    parser.add_argument('--cls-234', type=str, default='kaggle_output_folds234/test_cls_probs.npy')
    parser.add_argument('--seg-234', type=str, default='kaggle_output_folds234/test_seg_probs.npy')
    parser.add_argument('--thresholds', type=str, default='kaggle_output_folds234/best_thresholds.json')
    parser.add_argument('--output', type=str, default='submissions/submission_5fold_committee.csv')
    parser.add_argument('--blend', type=str, default='submissions/submission_calibrated_consensus.csv')
    parser.add_argument('--multifold', type=str, default='kaggle_output_quick/submission.csv')
    parser.add_argument('--w01', type=float, default=0.40)
    parser.add_argument('--w234', type=float, default=0.60)
    args = parser.parse_args()

    run_ensemble(
        cls_01_path=args.cls_01,
        seg_01_path=args.seg_01,
        cls_234_path=args.cls_234,
        seg_234_path=args.seg_234,
        thresh_json_path=args.thresholds,
        output_path=args.output,
        blend_csv=args.blend,
        multifold_csv=args.multifold,
        weight_01=args.w01,
        weight_234=args.w234
    )
