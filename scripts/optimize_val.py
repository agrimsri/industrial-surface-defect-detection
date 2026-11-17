import os
import sys
import json
import time
import numpy as np

if '.' not in sys.path:
    sys.path.append('.')

from src.postprocess import postprocess_mask

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

def evaluate_class_config(c, c_cls, c_seg, c_tgt, ct, pt, ms):
    """
    Evaluates a single class configuration over all validation samples.
    c_cls: (N,)
    c_seg: (N, H, W)
    c_tgt: (N, H, W)
    """
    N = len(c_cls)
    t_sums = c_tgt.reshape(N, -1).sum(axis=1)
    pos_idx = np.where(t_sums > 0)[0]
    neg_idx = np.where(t_sums == 0)[0]
    
    pos_dices = []
    neg_fp = []
    neg_emp = []
    pos_preds_count = 0
    
    for i in range(N):
        is_pos = (t_sums[i] > 0)
        p_cls = c_cls[i]
        
        if p_cls < ct:
            p_sum = 0
        else:
            mask = postprocess_mask(
                c_seg[i].astype(np.float32),
                cls_prob=p_cls,
                cls_idx=c,
                cls_thresh=ct,
                prob_thresh=pt,
                min_size=ms
            )
            p_sum = mask.sum()
            
        if p_sum > 0:
            pos_preds_count += 1
            
        if is_pos:
            if p_sum == 0:
                pos_dices.append(0.0)
            else:
                inter = np.logical_and(mask, c_tgt[i]).sum()
                pos_dices.append(float(2.0 * inter / (p_sum + t_sums[i])))
        else:
            if p_sum == 0:
                neg_fp.append(1.0)
                neg_emp.append(1.0)
            else:
                penalty = min(1.0, float(p_sum) / 100.0)
                neg_fp.append(max(0.0, 1.0 - penalty))
                neg_emp.append(0.0)
                
    s_dice = float(np.mean(pos_dices)) if pos_dices else 0.0
    s_fp = float(np.mean(neg_fp)) if neg_fp else 1.0
    s_emp = float(np.mean(neg_emp)) if neg_emp else 1.0
    score = 0.70 * s_dice + 0.20 * s_fp + 0.10 * s_emp
    
    return {
        'score': score,
        's_dice': s_dice,
        's_fp': s_fp,
        's_emp': s_emp,
        'pos_preds': pos_preds_count,
        'pos_rate': pos_preds_count / N,
        'true_pos': len(pos_idx),
        'true_rate': len(pos_idx) / N
    }

def main():
    cache_path = 'checkpoints/val_tta_cache.npz'
    if not os.path.exists(cache_path):
        print(f"Waiting for {cache_path}...")
        return
        
    data = np.load(cache_path)
    cls_probs = data['cls_probs']
    seg_probs = data['seg_probs']
    targets = data['targets']
    N = len(cls_probs)
    print(f"Loaded validation cache: N={N} samples")
    
    # 1. Evaluate prior attempt thresholds
    with open('kaggle_output/best_thresholds.json') as f:
        prior_cfg = json.load(f)
        
    print("\n" + "=" * 80)
    print("BASELINE: Evaluating Prior Attempt Thresholds")
    print("=" * 80)
    total_score = 0
    for c in range(4):
        ct = float(prior_cfg['cls_thresh'][str(c)])
        pt = float(prior_cfg['prob_thresh'][str(c)])
        ms = int(prior_cfg['min_size'][str(c)])
        res = evaluate_class_config(c, cls_probs[:, c], seg_probs[:, c], targets[:, c], ct, pt, ms)
        print(f"[{DEFECT_TYPES[c]:8s}] ct={ct:.2f}, pt={pt:.2f}, ms={ms:3d} -> Score={res['score']:.5f} "
              f"(Dice={res['s_dice']:.4f}, FP={res['s_fp']:.4f}, Emp={res['s_emp']:.4f}) | "
              f"Pred={res['pos_preds']}/{N} ({res['pos_rate']*100:.1f}%), True={res['true_pos']}/{N} ({res['true_rate']*100:.1f}%)")
        total_score += 0.25 * res['score']
    print(f"Overall Prior Thresholds Score: {total_score:.5f}")

if __name__ == '__main__':
    main()
