import cv2
import numpy as np
import pandas as pd

DEFAULT_CLS_THRESHOLDS = {0: 0.12, 1: 0.08, 2: 0.18, 3: 0.12}
DEFAULT_PROB_THRESHOLDS = {0: 0.45, 1: 0.40, 2: 0.50, 3: 0.45}
DEFAULT_MIN_SIZES = {0: 40, 1: 120, 2: 40, 3: 150}

def rle_decode(rle_str, shape=(128, 800)):
    """
    Decodes Run-Length Encoded string to binary mask.
    Index base: 1-indexed, Column-major (Fortran order).
    """
    if pd.isna(rle_str) or not str(rle_str).strip():
        return np.zeros(shape, dtype=np.uint8)
    s = str(rle_str).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def rle_encode(mask):
    """
    Encodes binary mask (128, 800) to Run-Length Encoded string.
    Index base: 1-indexed, Column-major (Fortran order).
    """
    dots = mask.flatten(order='F')
    if not np.any(dots):
        return ''
    runs = np.where(dots[1:] != dots[:-1])[0] + 2
    runs = np.concatenate([
        [1] if dots[0] else np.empty(0, dtype=int),
        runs,
        [len(dots) + 1] if dots[-1] else np.empty(0, dtype=int)
    ]).astype(int)
    starts = runs[::2]
    lengths = runs[1::2] - starts
    return ' '.join(f'{int(s)} {int(l)}' for s, l in zip(starts, lengths))

def postprocess_mask(prob_mask,
                     cls_prob=None,
                     cls_idx=0,
                     cls_thresh=None,
                     prob_thresh=None,
                     min_size=None,
                     apply_morphology=True):
    """
    Two-Stage Gated Post-Processing Pipeline:
    1. Classification Gating: If cls_prob < cls_thresh, output all-zero mask.
    2. Probability Thresholding: Binary mask = prob_mask > prob_thresh.
    3. Morphological Enhancement (defect-tailored).
    4. Component Filtering: Zero out components with area < min_size.
    
    Supports cls_thresh, prob_thresh, min_size as either scalar values or
    dictionaries mapping class index (int or str) to thresholds.
    """
    # Resolve per-class threshold from dict, scalar, or defaults
    if isinstance(cls_thresh, dict):
        cls_thresh = cls_thresh.get(cls_idx, cls_thresh.get(str(cls_idx), DEFAULT_CLS_THRESHOLDS.get(cls_idx, 0.50)))
    elif cls_thresh is None:
        cls_thresh = DEFAULT_CLS_THRESHOLDS.get(cls_idx, 0.50)

    if isinstance(prob_thresh, dict):
        prob_thresh = prob_thresh.get(cls_idx, prob_thresh.get(str(cls_idx), DEFAULT_PROB_THRESHOLDS.get(cls_idx, 0.50)))
    elif prob_thresh is None:
        prob_thresh = DEFAULT_PROB_THRESHOLDS.get(cls_idx, 0.50)

    if isinstance(min_size, dict):
        min_size = min_size.get(cls_idx, min_size.get(str(cls_idx), DEFAULT_MIN_SIZES.get(cls_idx, 50)))
    elif min_size is None:
        min_size = DEFAULT_MIN_SIZES.get(cls_idx, 50)

    # Stage 1: Classification Gating
    if cls_prob is not None and cls_prob < cls_thresh:
        return np.zeros(prob_mask.shape, dtype=np.uint8)

    # Stage 2: Probability Thresholding
    binary = (prob_mask > prob_thresh).astype(np.uint8)
    if not np.any(binary):
        return binary

    # Stage 3: Class-Specific Morphological Refinement
    if apply_morphology:
        if cls_idx == 1:
            # Defect B: Ultra-thin vertical scratches. Close vertical gaps.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif cls_idx == 3:
            # Defect D: Rolling abrasion bands. Dilate horizontally.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Stage 4: Connected Component Size Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean_mask[labels == lbl] = 1

    # Final overall size check
    if clean_mask.sum() < min_size:
        clean_mask.fill(0)

    return clean_mask

def enforce_mask_disjointness(masks, probs=None, class_priors=None, min_sizes=None):
    """
    Enforces strict cross-class disjointness across all 4 defect types for a single image.
    In ground truth, defects never overlap across classes (0 overlapping pixels).
    When ensembling or processing independent sigmoid outputs, resolves pixel overlaps by
    assigning each contested pixel to the class with highest predicted probability / confidence (argmax).

    Args:
        masks: np.ndarray of shape (C, H, W) binary masks (0 or 1), or list/dict of C masks.
        probs: optional np.ndarray of shape (C, H, W) continuous probabilities, soft
               consensus agreement, or joint confidence maps.
        class_priors: optional list or dict of length C with empirical class priors or weights
                      (e.g., [0.0949, 0.0275, 0.3599, 0.1498]).
        min_sizes: optional dict or list of minimum component sizes per class to re-filter
                   any tiny disconnected fragments caused by pixel carving.

    Returns:
        disjoint_masks: np.ndarray of shape (C, H, W) binary uint8 masks with strictly zero overlapping pixels.
    """
    if isinstance(masks, list):
        masks = np.stack(masks, axis=0)
    elif isinstance(masks, dict):
        masks = np.stack([masks[k] for k in sorted(masks.keys())], axis=0)

    masks = np.asarray(masks, dtype=np.uint8)
    C, H, W = masks.shape

    overlap_map = masks.sum(axis=0)
    contested = overlap_map > 1
    if not np.any(contested):
        return masks

    # Determine score map for contested pixels
    if probs is not None:
        probs_arr = np.asarray(probs, dtype=np.float32)
        if class_priors is not None:
            p_arr = np.array([class_priors[c] if not isinstance(class_priors, dict) else class_priors.get(c, class_priors.get(str(c), 1.0)) for c in range(C)], dtype=np.float32)
            probs_arr = probs_arr * p_arr[:, None, None]
        score_map = probs_arr * masks.astype(np.float32)
    else:
        default_p = np.array([0.0949, 0.0275, 0.3599, 0.1498], dtype=np.float32)
        p_arr = np.array([class_priors[c] if not isinstance(class_priors, dict) else class_priors.get(c, class_priors.get(str(c), default_p[c])) for c in range(C)], dtype=np.float32) if class_priors is not None else default_p
        score_map = p_arr[:, None, None] * masks.astype(np.float32)

    best_c = np.argmax(score_map, axis=0)

    disjoint_masks = masks.copy()
    for c in range(C):
        drop = contested & (best_c != c)
        disjoint_masks[c, drop] = 0

    # Re-filter connected components if min_sizes provided
    if min_sizes is not None:
        for c in range(C):
            ms = min_sizes.get(c, min_sizes.get(str(c), DEFAULT_MIN_SIZES.get(c, 50))) if isinstance(min_sizes, dict) else min_sizes[c]
            b = disjoint_masks[c]
            if not np.any(b):
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
            clean = np.zeros_like(b)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= ms:
                    clean[labels == lbl] = 1
            if clean.sum() < ms:
                clean.fill(0)
            disjoint_masks[c] = clean

    assert (disjoint_masks.sum(axis=0) <= 1).all(), "Disjointness enforcement failed to resolve all overlaps!"
    return disjoint_masks
