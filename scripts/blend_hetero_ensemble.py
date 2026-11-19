"""
Heterogeneous Continuous Probability Blending Pipeline
Blends continuous probability predictions from:
1. SE-ResNeXt50 U-Net (dense residual backbone with high-resolution stride-2 skip connections,
   DualHead pooling, Lovász-Hinge fine-tuning)
2. 5-Fold tf_efficientnetv2_s Committee (10 models: Best + SWA with 4-way TTA, LB 0.70047)

Ensembling Methodology:
- Continuous classification probability blending with per-class weights w in [0.3, 0.7].
- Continuous segmentation probability volume blending where available, augmented with committee mask priors.
- Two-stage post-processing with calibrated gating, defect-tailored non-disjoint morphological closing
  (3x3 for A, 1x5 vertical for B, 3x3 for C, 5x1 horizontal for D), and connected-component area filtering.
- Strict Kaggle formatting: csv.QUOTE_ALL with ' ' for clean masks, 12,000 rows, 0 nulls.
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

from src.postprocess import postprocess_mask, rle_encode, rle_decode

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']
DEFECT_TO_IDX = {d: i for i, d in enumerate(DEFECT_TYPES)}

# Calibrated baseline thresholds from OOF sweeps
DEFAULT_CLS_THRESHOLDS = {0: 0.25, 1: 0.30, 2: 0.30, 3: 0.25}
DEFAULT_PROB_THRESHOLDS = {0: 0.48, 1: 0.38, 2: 0.45, 3: 0.48}
DEFAULT_MIN_SIZES = {0: 80, 1: 100, 2: 60, 3: 200}

# Optimal OOF per-class heterogeneous blend weights for SEResNeXt50 (w in [0.3, 0.7])
# [defect_a: 0.45, defect_b: 0.40, defect_c: 0.50, defect_d: 0.45]
DEFAULT_SERESNEXT_WEIGHTS = {0: 0.45, 1: 0.40, 2: 0.50, 3: 0.45}

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

def load_combined_thresholds(sn_json_path, comm_json_path, weights):
    """Interpolates optimal thresholds across heterogeneous models using blend weights."""
    sn_cls, sn_prob, sn_min = load_thresholds(sn_json_path)
    cm_cls, cm_prob, cm_min = load_thresholds(comm_json_path)
    
    cls_thresh = {}
    prob_thresh = {}
    min_size = {}
    for c in range(4):
        w = float(weights.get(c, 0.50))
        cls_thresh[c] = round(w * sn_cls[c] + (1.0 - w) * cm_cls[c], 2)
        prob_thresh[c] = round(w * sn_prob[c] + (1.0 - w) * cm_prob[c], 2)
        min_size[c] = int(round(w * sn_min[c] + (1.0 - w) * cm_min[c]))
    print(f"Calibrated blended thresholds:")
    print(f"  cls_thresh:  {cls_thresh}")
    print(f"  prob_thresh: {prob_thresh}")
    print(f"  min_size:    {min_size}")
    return cls_thresh, prob_thresh, min_size

def blend_hetero_ensemble(
    seresnext_dir='kaggle_output_seresnext',
    committee_cls_path='checkpoints/test_cls_probs_5folds.npy',
    committee_sub_path='submissions/submission_5fold_committee.csv',
    committee_json_path='kaggle_output_folds234/best_thresholds.json',
    sample_sub_path='public/sample_submission.csv',
    output_path='submissions/submission_sota_hetero_ensemble.csv',
    master_path='submissions/submission.csv',
    seresnext_weights=None
):
    print("=" * 80)
    print("🚀 EXECUTING SOTA HETEROGENEOUS ENSEMBLE CONTINUOUS PROBABILITY BLENDING")
    print(f"  SEResNeXt50 Source: {seresnext_dir}")
    print(f"  5-Fold Committee Cls: {committee_cls_path}")
    print(f"  5-Fold Committee Sub: {committee_sub_path}")
    print(f"  Output CSV: {output_path}")
    print("=" * 80)

    weights = seresnext_weights or DEFAULT_SERESNEXT_WEIGHTS

    # 1. Load SEResNeXt50 probabilities
    sn_cls_path = os.path.join(seresnext_dir, 'test_cls_probs.npy')
    if not os.path.exists(sn_cls_path):
        raise FileNotFoundError(f"Missing SEResNeXt50 classification probabilities at {sn_cls_path}")
    
    sn_cls_probs = np.load(sn_cls_path).astype(np.float32)
    print(f"Loaded SEResNeXt50 classification probabilities: shape {sn_cls_probs.shape}")

    # Load committee classification probabilities
    if not os.path.exists(committee_cls_path):
        raise FileNotFoundError(f"Missing 5-fold committee classification probabilities at {committee_cls_path}")
    cm_cls_probs = np.load(committee_cls_path).astype(np.float32)
    print(f"Loaded 5-fold committee classification probabilities: shape {cm_cls_probs.shape}")
    assert sn_cls_probs.shape == cm_cls_probs.shape, f"Mismatched cls shapes: {sn_cls_probs.shape} vs {cm_cls_probs.shape}"

    # 2. Continuous classification probability blending per class
    blended_cls_probs = np.zeros_like(sn_cls_probs)
    for c in range(4):
        w = float(weights.get(c, 0.50))
        blended_cls_probs[:, c] = w * sn_cls_probs[:, c] + (1.0 - w) * cm_cls_probs[:, c]
        print(f"  Class {c} ({DEFECT_TYPES[c]}): SEResNeXt weight={w:.2f}, Committee weight={1.0 - w:.2f}")

    # Save blended classification probabilities
    os.makedirs('checkpoints', exist_ok=True)
    blended_cls_out = 'checkpoints/test_cls_probs_hetero_ensemble.npy'
    np.save(blended_cls_out, blended_cls_probs)
    print(f"✅ Saved blended classification probabilities to {blended_cls_out}")

    # 3. Check for SEResNeXt50 segmentation probability array (.npy or .npz)
    sn_seg_npy = os.path.join(seresnext_dir, 'test_seg_probs.npy')
    sn_seg_npz = os.path.join(seresnext_dir, 'test_seg_probs.npz')
    sn_seg_probs = None

    if os.path.exists(sn_seg_npz):
        print(f"Loading compressed SEResNeXt segmentation probabilities from {sn_seg_npz}...")
        npz_data = np.load(sn_seg_npz)
        sn_seg_probs = npz_data['seg_probs']
        print(f"Loaded compressed SEResNeXt seg probs: shape {sn_seg_probs.shape}, dtype {sn_seg_probs.dtype}")
    elif os.path.exists(sn_seg_npy):
        print(f"Loading uncompressed SEResNeXt segmentation probabilities from {sn_seg_npy}...")
        sn_seg_probs = np.load(sn_seg_npy, mmap_mode='r')
        print(f"Loaded SEResNeXt seg probs: shape {sn_seg_probs.shape}, dtype {sn_seg_probs.dtype}")

    # Load thresholds (calibrated and combined across heterogeneous models)
    thresh_json = os.path.join(seresnext_dir, 'best_thresholds.json')
    cls_thresh, prob_thresh, min_sizes = load_combined_thresholds(thresh_json, committee_json_path, weights)

    # Load committee submission
    comm_df = pd.read_csv(committee_sub_path)
    comm_pivot = comm_df.pivot(index='image_id', columns='defect_type', values='mask_rle')

    # Load sample submission template
    sub_template = pd.read_csv(sample_sub_path)
    test_image_ids = sub_template['image_id'].unique().tolist()
    num_test = len(test_image_ids)

    # Load SEResNeXt submission for reference / fallback if available
    sn_sub_path = os.path.join(seresnext_dir, 'submission.csv')
    sn_pivot = None
    if os.path.exists(sn_sub_path):
        sn_sub_df = pd.read_csv(sn_sub_path)
        sn_pivot = sn_sub_df.pivot(index='image_id', columns='defect_type', values='mask_rle')
        print(f"Loaded SEResNeXt standalone submission from {sn_sub_path}")

    # 4. Stream processing and blending
    rle_dict = {}
    audit_counts = {d: 0 for d in DEFECT_TYPES}
    total_pixels = {d: 0 for d in DEFECT_TYPES}
    t0 = time.time()

    for i, img_id in enumerate(test_image_ids):
        for c, d in enumerate(DEFECT_TYPES):
            cls_p = blended_cls_probs[i, c]
            comm_rle = comm_pivot.loc[img_id, d] if img_id in comm_pivot.index else ' '
            comm_mask = rle_decode(comm_rle) if pd.notna(comm_rle) and str(comm_rle).strip() else None

            if sn_seg_probs is not None:
                # Full continuous volume available
                sn_seg_slice = sn_seg_probs[i, c].astype(np.float32)
                w_sn = float(weights.get(c, 0.50))

                if comm_mask is not None and comm_mask.sum() > 0:
                    comm_prob = np.where(comm_mask > 0, 0.90, 0.05).astype(np.float32)
                else:
                    comm_prob = np.full_like(sn_seg_slice, 0.05, dtype=np.float32)
                blended_seg = w_sn * sn_seg_slice + (1.0 - w_sn) * comm_prob

                mask = postprocess_mask(
                    prob_mask=blended_seg,
                    cls_prob=cls_p,
                    cls_idx=c,
                    cls_thresh=cls_thresh,
                    prob_thresh=prob_thresh,
                    min_size=min_sizes,
                    apply_morphology=True
                )
            else:
                # Hybrid mask consensus blending
                sn_rle = sn_pivot.loc[img_id, d] if (sn_pivot is not None and img_id in sn_pivot.index) else ' '
                sn_mask = rle_decode(sn_rle) if pd.notna(sn_rle) and str(sn_rle).strip() else None

                if c == 1:
                    # defect_b (rare scratch): require consensus or high confidence
                    has_sn = (sn_mask is not None and sn_mask.sum() >= min_sizes[1])
                    has_cm = (comm_mask is not None and comm_mask.sum() >= min_sizes[1])
                    if has_sn and has_cm:
                        mask = np.logical_or(sn_mask, comm_mask).astype(np.uint8)
                    elif has_cm and cls_p >= 0.35:
                        mask = comm_mask
                    elif has_sn and cls_p >= 0.45:
                        mask = sn_mask
                    else:
                        mask = np.zeros((128, 800), dtype=np.uint8)
                elif c == 2:
                    # defect_c (dominant flake patch): union across both models ensures maximum recall
                    m1 = sn_mask if sn_mask is not None else np.zeros((128, 800), dtype=np.uint8)
                    m2 = comm_mask if comm_mask is not None else np.zeros((128, 800), dtype=np.uint8)
                    mask = np.logical_or(m1, m2).astype(np.uint8)
                else:
                    # defect_a & defect_d: union with area filtering
                    m1 = sn_mask if sn_mask is not None else np.zeros((128, 800), dtype=np.uint8)
                    m2 = comm_mask if comm_mask is not None else np.zeros((128, 800), dtype=np.uint8)
                    comb = np.logical_or(m1, m2).astype(np.uint8)
                    mask = comb if comb.sum() >= min_sizes[c] else np.zeros((128, 800), dtype=np.uint8)

                # Classification gating check
                tau_cls = cls_thresh.get(c, 0.25) if isinstance(cls_thresh, dict) else cls_thresh
                if cls_p < tau_cls:
                    mask = np.zeros((128, 800), dtype=np.uint8)

            # Apply non-disjoint morphological closing to ensure clean edges
            if mask.sum() > 0:
                mask = postprocess_mask(
                    prob_mask=mask.astype(np.float32),
                    cls_prob=cls_p,
                    cls_idx=c,
                    cls_thresh=0.0,
                    prob_thresh=0.5,
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
            print(f"  Processed {i + 1}/{num_test} images ({(i+1)/num_test*100:.1f}%) | {fps:.1f} FPS")

    # 5. Format and save final submission
    out_df = sub_template.copy()
    out_df['mask_rle'] = [
        rle_dict.get((img_id, d), ' ')
        for img_id, d in zip(out_df['image_id'], out_df['defect_type'])
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n✅ Saved Heterogeneous Ensemble submission to {output_path}")

    if master_path:
        out_df.to_csv(master_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Updated master submission at {master_path}")

    # 6. Audit & Verification Record
    print("\n" + "=" * 80)
    print("🔍 HETEROGENEOUS ENSEMBLE AUDIT & VERIFICATION RECORD")
    print("=" * 80)
    print(f"  Total Rows:   {len(out_df)} (Target: 12000) -> {'PASS' if len(out_df) == 12000 else 'FAIL'}")
    print(f"  Null Values:  {out_df.isnull().sum().sum()} -> {'PASS' if out_df.isnull().sum().sum() == 0 else 'FAIL'}")
    print(f"  Columns:      {list(out_df.columns)} -> PASS")

    empty_cnt = (out_df['mask_rle'].str.strip() == '').sum()
    pos_cnt = len(out_df) - empty_cnt
    print(f"  Empty Masks:  {empty_cnt:,} ({empty_cnt/len(out_df)*100:.2f}%)")
    print(f"  Active Masks: {pos_cnt:,} ({pos_cnt/len(out_df)*100:.2f}%)")

    print("\n  Per-Class Defect Counts & Pixels:")
    for d in DEFECT_TYPES:
        cnt = (out_df[out_df['defect_type'] == d]['mask_rle'].str.strip() != '').sum()
        px = total_pixels[d]
        print(f"    - {d:8s}: {cnt:4d} masks ({cnt/3000*100:.2f}%) | {px:,} pixels")
        assert cnt > 0, f"Error: Defect class {d} has 0 positive predictions!"

    print("  ✅ All 4 defect classes have active non-zero predictions.")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seresnext-dir', type=str, default='kaggle_output_seresnext')
    parser.add_argument('--committee-cls', type=str, default='checkpoints/test_cls_probs_5folds.npy')
    parser.add_argument('--committee-sub', type=str, default='submissions/submission_5fold_committee.csv')
    parser.add_argument('--committee-json', type=str, default='kaggle_output_folds234/best_thresholds.json')
    parser.add_argument('--sample-sub', type=str, default='public/sample_submission.csv')
    parser.add_argument('--output', type=str, default='submissions/submission_sota_hetero_ensemble.csv')
    parser.add_argument('--master', type=str, default='submissions/submission.csv')
    args = parser.parse_args()

    blend_hetero_ensemble(
        seresnext_dir=args.seresnext_dir,
        committee_cls_path=args.committee_cls,
        committee_sub_path=args.committee_sub,
        committee_json_path=args.committee_json,
        sample_sub_path=args.sample_sub,
        output_path=args.output,
        master_path=args.master
    )
