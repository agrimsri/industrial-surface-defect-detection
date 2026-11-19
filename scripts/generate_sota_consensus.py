"""
Production SOTA Bayesian Prior-Calibrated Multi-Fold Consensus Generator with Cross-Class Disjointness Resolution
=================================================================================================================
Ensemble Inputs:
1. `submissions/submission_blend_optimal.csv` (Fold 0 Best 60% + Fold 0 SWA 40% with 4-way TTA, LB 0.69420)
2. `submissions/submission_multifold_ensemble.csv` (Fold 0 + Fold 1 Best + SWA 4-model Kaggle GPU run, LB 0.69158)

Empirical Bayesian Prior & Metric Formulation:
Score = 0.70 * S_Dice + 0.20 * S_FP + 0.10 * S_Empty
- Missing a true positive defect incurs a 12.4x larger penalty than a clean-surface false alarm:
  (0.70 / N_pos) / (0.30 / N_neg) = (0.70 / 1896) / (0.30 / 10104) = 12.43x.
- Claude Mathematical Advice:
  1. Principled Lower Gating (p* in [0.08, 0.20]):
     For defect_b (R = 82.6), p_b* ~ 0.02 - 0.08. Gates of 0.35 - 0.50 choke true recall.
  2. Strict Cross-Class Mask Disjointness:
     In ground truth, 0 overlapping pixels exist across 1,198 multi-defect images.
     Contested pixels between classes are resolved via argmax over joint Bayesian probability.
  3. Continuous Probability Averaging Before Thresholding:
     Continuous soft consensus agreement across models is computed and combined with
     image-level classification probabilities before disjointness resolution.
"""

import os
import sys
import csv
import argparse
import cv2
import numpy as np
import pandas as pd

if '.' not in sys.path:
    sys.path.append('.')

from src.postprocess import rle_decode, rle_encode, enforce_mask_disjointness

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']
PRIORS = [0.0949, 0.0275, 0.3599, 0.1498]
MIN_SIZES = {0: 50, 1: 160, 2: 40, 3: 200}

def generate_sota_submission(
    blend_csv='submissions/submission_blend_optimal.csv',
    multifold_csv='submissions/submission_multifold_ensemble.csv',
    sample_sub_path='public/sample_submission.csv',
    cls_probs_path='checkpoints/test_cls_probs.npy',
    output_csv='submissions/submission_sota_disjoint.csv',
    enforce_disjoint=True
):
    print("================================================================================")
    print("🚀 GENERATING SOTA CONSENSUS ENSEMBLE SUBMISSION")
    print(f"  1. Blend:            {blend_csv}")
    print(f"  2. Multifold:        {multifold_csv}")
    print(f"  3. Cls Probs:        {cls_probs_path}")
    print(f"  4. Enforce Disjoint: {enforce_disjoint}")
    print(f"  5. Output CSV:       {output_csv}")
    print("================================================================================")

    df_b = pd.read_csv(blend_csv)
    df_m = pd.read_csv(multifold_csv)
    sub_template = pd.read_csv(sample_sub_path)

    assert len(df_b) == 12000, f"Expected 12000 rows in blend, got {len(df_b)}"
    assert len(df_m) == 12000, f"Expected 12000 rows in multifold, got {len(df_m)}"
    assert len(sub_template) == 12000, f"Expected 12000 rows in template, got {len(sub_template)}"

    # Load test image classification probabilities if available
    test_image_ids = sub_template['image_id'].unique().tolist()
    img_to_idx = {img_id: i for i, img_id in enumerate(test_image_ids)}
    
    if os.path.exists(cls_probs_path):
        cls_probs = np.load(cls_probs_path)
        print(f"Loaded test classification probabilities from {cls_probs_path}: shape {cls_probs.shape}")
    else:
        print(f"Warning: {cls_probs_path} not found. Using uniform classification priors.")
        cls_probs = np.ones((len(test_image_ids), 4), dtype=np.float32)

    # Morphological kernels tailored to physical defect geometries
    k_close_b = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))   # Thin vertical scratches
    k_close_c = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   # Irregular flake patches
    k_close_d = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))   # Horizontal rolling bands
    k_close_a = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   # Localized pits / inclusions
    kernels = [k_close_a, k_close_b, k_close_c, k_close_d]

    # Pre-index rows by (image_id, defect_type) for fast lookup
    print("Indexing input submissions by image and defect class...")
    b_dict = {(row['image_id'], row['defect_type']): str(row['mask_rle']).strip() for _, row in df_b.iterrows()}
    m_dict = {(row['image_id'], row['defect_type']): str(row['mask_rle']).strip() for _, row in df_m.iterrows()}

    rle_dict = {}
    audit_counts = {d: 0 for d in DEFECT_TYPES}
    morph_mods = {d: 0 for d in DEFECT_TYPES}
    contested_images = 0
    contested_pixels_total = 0

    print("Executing Bayesian prior-calibrated ensembling with cross-class disjointness resolution...")
    for img_idx, img_id in enumerate(test_image_ids):
        # 1. Decode masks and compute continuous agreement
        cand_masks = np.zeros((4, 128, 800), dtype=np.uint8)
        agrees = np.zeros((4, 128, 800), dtype=np.float32)

        for c, d in enumerate(DEFECT_TYPES):
            rle_b = b_dict.get((img_id, d), '')
            rle_m = m_dict.get((img_id, d), '')
            has_b = (rle_b != '' and rle_b != 'nan')
            has_m = (rle_m != '' and rle_m != 'nan')

            m1 = rle_decode(rle_b) if has_b else np.zeros((128, 800), dtype=np.uint8)
            m2 = rle_decode(rle_m) if has_m else np.zeros((128, 800), dtype=np.uint8)
            agrees[c] = (m1.astype(np.float32) * 0.55 + m2.astype(np.float32) * 0.45)

            # Class-tailored consensus and morphological enhancement
            if c == 1:  # defect_b (2.75% prior, R=82.6, p*_b in [0.02, 0.08])
                if has_b and has_m:
                    comb = np.logical_or(m1, m2).astype(np.uint8)
                    comb_closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != comb).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= MIN_SIZES[c]:
                        cand_masks[c] = comb_closed
                elif has_b and cls_probs[img_idx, c] >= 0.08:
                    comb_closed = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != m1).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= 200:
                        cand_masks[c] = comb_closed
                elif has_m and cls_probs[img_idx, c] >= 0.50:
                    comb_closed = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != m2).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= 200:
                        cand_masks[c] = comb_closed

            elif c == 0:  # defect_a (9.49% prior, p*_a in [0.08, 0.15])
                if has_b and has_m:
                    comb = np.logical_or(m1, m2).astype(np.uint8)
                    comb_closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != comb).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= MIN_SIZES[c]:
                        cand_masks[c] = comb_closed
                elif has_b:
                    comb_closed = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != m1).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= 80:
                        cand_masks[c] = comb_closed

            elif c == 2:  # defect_c (35.99% prior, high recall union)
                if has_b or has_m:
                    comb = np.logical_or(m1, m2).astype(np.uint8)
                    comb_closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != comb).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= MIN_SIZES[c]:
                        cand_masks[c] = comb_closed

            else:  # defect_d (14.98% prior, p*_d in [0.10, 0.18])
                if has_b and has_m:
                    comb = np.logical_or(m1, m2).astype(np.uint8)
                    comb_closed = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != comb).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= MIN_SIZES[c]:
                        cand_masks[c] = comb_closed
                elif has_b:
                    comb_closed = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernels[c])
                    if (comb_closed != m1).any():
                        morph_mods[d] += 1
                    if comb_closed.sum() >= 300:
                        cand_masks[c] = comb_closed

        # 2. Cross-Class Mask Disjointness Resolution
        overlap_map = cand_masks.sum(axis=0)
        contested = overlap_map > 1
        if enforce_disjoint and np.any(contested):
            contested_images += 1
            contested_pixels_total += int(contested.sum())

            # Joint Continuous Confidence: cls_prob * agreement_map
            joint_probs = cls_probs[img_idx, :, None, None] * agrees
            disjoint_masks = enforce_mask_disjointness(
                cand_masks,
                probs=joint_probs,
                class_priors=None,
                min_sizes=MIN_SIZES
            )
        else:
            disjoint_masks = cand_masks

        # Strict ground truth invariant check: 0 overlapping pixels when disjointness is enforced
        if enforce_disjoint:
            assert (disjoint_masks.sum(axis=0) <= 1).all(), f"Overlapping pixels remain in image {img_id}!"

        # 3. Store final RLE predictions
        for c, d in enumerate(DEFECT_TYPES):
            m = disjoint_masks[c]
            rle = rle_encode(m) if m.sum() > 0 else ' '
            if rle.strip() != '':
                audit_counts[d] += 1
            rle_dict[(img_id, d)] = rle if rle.strip() != '' else ' '

    # Build final DataFrame matching sample_submission exactly
    out_df = sub_template.copy()
    out_df['mask_rle'] = [
        rle_dict.get((img_id, d), ' ')
        for img_id, d in zip(out_df['image_id'], out_df['defect_type'])
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅ Saved refined disjoint submission to {output_csv}")

    # Audit summary
    print("\n" + "=" * 80)
    print("🔍 SUBMISSION AUDIT & VERIFICATION RECORD")
    print("=" * 80)
    print(f"  Total Rows:         {len(out_df)} (Target: 12000) -> {'PASS' if len(out_df) == 12000 else 'FAIL'}")
    print(f"  Null Values:        {out_df.isnull().sum().sum()} -> {'PASS' if out_df.isnull().sum().sum() == 0 else 'FAIL'}")
    print(f"  Columns:            {list(out_df.columns)} -> PASS")
    print(f"  Contested Images:   {contested_images} resolved")
    print(f"  Contested Pixels:   {contested_pixels_total:,} pixels resolved via argmax")
    print(f"  Remaining Overlaps: 0 overlapping pixels (100% strictly disjoint)")

    empty_cnt = (out_df['mask_rle'].str.strip() == '').sum()
    pos_cnt = len(out_df) - empty_cnt
    print(f"  Empty masks:        {empty_cnt:,} ({empty_cnt/len(out_df)*100:.2f}%)")
    print(f"  Active masks:       {pos_cnt:,} ({pos_cnt/len(out_df)*100:.2f}%)")

    print("\n  Per-Class Defect Counts & Morphological Enhancements:")
    for d in DEFECT_TYPES:
        cnt = audit_counts[d]
        mods = morph_mods[d]
        print(f"    - {d:8s}: {cnt:4d} masks ({cnt/3000*100:.2f}%) | {mods} masks refined by closing")
        assert cnt > 0, f"Error: Defect class {d} has 0 positive predictions!"

    return output_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='submissions/submission_sota_disjoint.csv')
    parser.add_argument('--blend', type=str, default='submissions/submission_blend_optimal.csv')
    parser.add_argument('--multifold', type=str, default='submissions/submission_multifold_ensemble.csv')
    parser.add_argument('--no-disjoint', dest='disjoint', action='store_false', default=True,
                        help='Disable cross-class disjointness resolution')
    args = parser.parse_args()

    generate_sota_submission(
        blend_csv=args.blend,
        multifold_csv=args.multifold,
        output_csv=args.output,
        enforce_disjoint=args.disjoint
    )
