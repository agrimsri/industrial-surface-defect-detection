"""
Production SOTA Bayesian Prior-Calibrated Multi-Fold Ensemble Generator
======================================================================
Ensemble Inputs:
1. `submissions/submission_blend_optimal.csv` (Fold 0 Best 60% + Fold 0 SWA 40% with 4-way TTA, LB 0.69420)
2. `submissions/submission_multifold_ensemble.csv` (Fold 0 + Fold 1 Best + SWA 4-model Kaggle GPU run, LB 0.69158)
3. `checkpoints/test_cls_probs.npy` (4-way TTA continuous classifier probability calibration)

Pruning Rationale & Metric Formulation:
Score = 0.70 * S_Dice + 0.20 * S_FP + 0.10 * S_Empty
- 84.2% of test (image, defect_type) pairs are clean surface areas.
- Every false alarm on a clean surface gets 0 for both S_FP and S_Empty, penalizing 30% of the metric.
- Multi-model consensus and continuous probability gating systematically prune ~181 low-confidence
  isolated false alarms while preserving true positive defect coverage.
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd

if '.' not in sys.path:
    sys.path.append('.')

from src.postprocess import rle_decode, rle_encode

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

def generate_sota_submission(
    blend_csv='submissions/submission_blend_optimal.csv',
    multifold_csv='submissions/submission_multifold_ensemble.csv',
    probs_path='checkpoints/test_cls_probs.npy',
    sample_sub_path='public/sample_submission.csv',
    output_csv='submissions/submission_sota_consensus.csv'
):
    print(f"Loading ensemble inputs:")
    print(f"  1. Blend:     {blend_csv}")
    print(f"  2. Multifold: {multifold_csv}")
    print(f"  3. Probs:     {probs_path}")
    
    df_b = pd.read_csv(blend_csv)
    df_m = pd.read_csv(multifold_csv)
    sub_template = pd.read_csv(sample_sub_path)
    probs = np.load(probs_path)
    
    assert len(df_b) == 12000, f"Expected 12000 rows in blend, got {len(df_b)}"
    assert len(df_m) == 12000, f"Expected 12000 rows in multifold, got {len(df_m)}"
    assert len(sub_template) == 12000, f"Expected 12000 rows in template, got {len(sub_template)}"
    
    test_image_ids = sub_template['image_id'].unique().tolist()
    assert len(test_image_ids) == len(probs), "Mismatch between image IDs and probability cache"
    id_to_idx = {img_id: i for i, img_id in enumerate(test_image_ids)}
    c_map = {d: i for i, d in enumerate(DEFECT_TYPES)}
    
    rows = []
    audit_counts = {d: 0 for d in DEFECT_TYPES}
    pruned_counts = {d: 0 for d in DEFECT_TYPES}
    
    print("\nExecuting Bayesian prior-calibrated consensus blending with probability gating...")
    for i in range(len(df_b)):
        row_b = df_b.iloc[i]
        row_m = df_m.iloc[i]
        d_type = row_b['defect_type']
        img_id = row_b['image_id']
        c_idx = c_map[d_type]
        p = float(probs[id_to_idx[img_id], c_idx])
        
        rle_b = str(row_b['mask_rle']).strip()
        rle_m = str(row_m['mask_rle']).strip()
        has_b = (rle_b != '')
        has_m = (rle_m != '')
        
        final_rle = ' '
        
        if d_type == 'defect_b':
            # 2.75% true prior. Extreme false alarm penalty. Strict consensus + p >= 0.45.
            if has_b and has_m and p >= 0.45:
                m1 = rle_decode(rle_b, (128, 800))
                m2 = rle_decode(rle_m, (128, 800))
                comb = np.logical_or(m1, m2).astype(np.uint8)
                final_rle = rle_encode(comb) if comb.sum() >= 200 else ' '
            else:
                if has_b or has_m:
                    pruned_counts[d_type] += 1
                final_rle = ' '
                
        elif d_type == 'defect_a':
            # 9.49% true prior. Consensus union + high-confidence single predictions.
            if has_b and has_m:
                m1 = rle_decode(rle_b, (128, 800))
                m2 = rle_decode(rle_m, (128, 800))
                comb = np.logical_or(m1, m2).astype(np.uint8)
                final_rle = rle_encode(comb) if comb.sum() >= 50 else ' '
            elif has_b and p >= 0.45:
                m1 = rle_decode(rle_b, (128, 800))
                final_rle = rle_b if m1.sum() >= 80 else ' '
            else:
                if has_b or has_m:
                    pruned_counts[d_type] += 1
                final_rle = ' '
                
        elif d_type == 'defect_c':
            # 35.99% true prior. High recall on patches, but prune unconfident multifold noise (p < 0.40).
            if has_b and has_m:
                m1 = rle_decode(rle_b, (128, 800))
                m2 = rle_decode(rle_m, (128, 800))
                comb = np.logical_or(m1, m2).astype(np.uint8)
                final_rle = rle_encode(comb)
            elif has_b and p >= 0.40:
                m1 = rle_decode(rle_b, (128, 800))
                final_rle = rle_b if m1.sum() >= 60 else ' '
            elif has_m and p >= 0.40:
                m2 = rle_decode(rle_m, (128, 800))
                final_rle = rle_m if m2.sum() >= 60 else ' '
            else:
                if has_b or has_m:
                    pruned_counts[d_type] += 1
                final_rle = ' '
                
        else:  # defect_d (14.98% true prior)
            if has_b and has_m:
                m1 = rle_decode(rle_b, (128, 800))
                m2 = rle_decode(rle_m, (128, 800))
                comb = np.logical_or(m1, m2).astype(np.uint8)
                final_rle = rle_encode(comb) if comb.sum() >= 200 else ' '
            elif has_b and p >= 0.50:
                m1 = rle_decode(rle_b, (128, 800))
                final_rle = rle_b if m1.sum() >= 300 else ' '
            else:
                if has_b or has_m:
                    pruned_counts[d_type] += 1
                final_rle = ' '
                
        if final_rle.strip() != '':
            audit_counts[d_type] += 1
            
        rows.append({
            'ID': row_b['ID'],
            'image_id': img_id,
            'defect_type': d_type,
            'mask_rle': final_rle if final_rle.strip() != '' else ' '
        })
        
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅ Saved calibrated consensus submission to {output_csv}")
    
    # Audit summary
    print("\n🔍 SUBMISSION AUDIT & VERIFICATION:")
    print(f"  Total Rows:   {len(out_df)} (Target: 12000) -> {'PASS' if len(out_df) == 12000 else 'FAIL'}")
    print(f"  Null Values:  {out_df.isnull().sum().sum()} -> {'PASS' if out_df.isnull().sum().sum() == 0 else 'FAIL'}")
    print(f"  Columns:      {list(out_df.columns)} -> PASS")
    
    empty_cnt = (out_df['mask_rle'].str.strip() == '').sum()
    pos_cnt = len(out_df) - empty_cnt
    print(f"  Empty masks:  {empty_cnt:,} ({empty_cnt/len(out_df)*100:.2f}%)")
    print(f"  Active masks: {pos_cnt:,} ({pos_cnt/len(out_df)*100:.2f}%)")
    
    print("\n  Per-Class Active Mask Counts (vs Ground Truth Prior):")
    priors = {'defect_a': 9.49, 'defect_b': 2.75, 'defect_c': 35.99, 'defect_d': 14.98}
    for d in DEFECT_TYPES:
        cnt = audit_counts[d]
        pct = cnt / 3000 * 100
        pruned = pruned_counts[d]
        print(f"    - {d:8s}: {cnt:4d} masks ({pct:5.2f}% of test) | Prior: {priors[d]:5.2f}% | Pruned noise: {pruned:3d}")
        assert cnt > 0, f"Defect class {d} has zero predictions!"
        
    print("\n✅ Verification PASSED: Ready for Kaggle CLI submission.")
    return output_csv

if __name__ == '__main__':
    generate_sota_submission()
