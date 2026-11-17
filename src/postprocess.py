import cv2
import numpy as np
import pandas as pd

DEFAULT_CLS_THRESHOLDS = {0: 0.50, 1: 0.35, 2: 0.50, 3: 0.45}
DEFAULT_PROB_THRESHOLDS = {0: 0.50, 1: 0.40, 2: 0.50, 3: 0.45}
DEFAULT_MIN_SIZES = {0: 50, 1: 95, 2: 40, 3: 100}

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
    """
    if cls_thresh is None:
        cls_thresh = DEFAULT_CLS_THRESHOLDS.get(cls_idx, 0.50)
    if prob_thresh is None:
        prob_thresh = DEFAULT_PROB_THRESHOLDS.get(cls_idx, 0.50)
    if min_size is None:
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
