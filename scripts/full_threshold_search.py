import os
import sys
import json
import time
import numpy as np

if '.' not in sys.path:
    sys.path.append('.')

from src.postprocess import postprocess_mask

DEFECT_TYPES = ['defect_a', 'defect_b', 'defect_c', 'defect_d']

def run_search():
    data = np.load('checkpoints/val_tta_cache.npz')
    cls_probs = data['cls_probs']
    seg_probs = data['seg_probs']
    targets = data['targets']
    N = len(cls_probs)
    
    # Ground truth statistics
    # defect_a: true rate ~ 9.1%
    # defect_b: true rate ~ 3.0%
    # defect_c: true rate ~ 36.4%
    # defect_d: true rate ~ 15.1%
    
    search_grids = {
        0: {  # defect_a
            'cls': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
            'prob': [0.45, 0.50, 0.55, 0.60],
            'min_size': [40, 60, 80, 100, 150]
        },
        1: {  # defect_b
            'cls': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            'prob': [0.35, 0.40, 0.45, 0.50],
            'min_size': [80, 100, 130, 160, 200]
        },
        2: {  # defect_c
            'cls': [0.35, 0.40, 0.45, 0.50, 0.55],
            'prob': [0.35, 0.40, 0.45, 0.50],
            'min_size': [40, 60, 80, 120, 160]
        },
        3: {  # defect_d
            'cls': [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
            'prob': [0.40, 0.45, 0.50, 0.55],
            'min_size': [60, 100, 150, 200, 300]
        }
    }
    
    best_results = {}
    
    for c in range(4):
        c_name = DEFECT_TYPES[c]
        c_cls = cls_probs[:, c]
        c_seg = seg_probs[:, c]
        c_tgt = targets[:, c]
        
        t_sums = c_tgt.reshape(N, -1).sum(axis=1)
        pos_indices = np.where(t_sums > 0)[0]
        neg_indices = np.where(t_sums == 0)[0]
        true_pos = len(pos_indices)
        
        grid = search_grids[c]
        print(f"\nSearching class {c_name} (True Positives: {true_pos}/{N} = {true_pos/N*100:.1f}%)...")
        
        top_configs = []
        
        for ct in grid['cls']:
            for pt in grid['prob']:
                for ms in grid['min_size']:
                    pos_dices = []
                    neg_fp = []
                    neg_emp = []
                    pred_pos_count = 0
                    
                    for idx in pos_indices:
                        t_sum = t_sums[idx]
                        p_cls = c_cls[idx]
                        if p_cls < ct:
                            pos_dices.append(0.0)
                        else:
                            mask = postprocess_mask(
                                c_seg[idx].astype(np.float32),
                                cls_prob=p_cls,
                                cls_idx=c,
                                cls_thresh=ct,
                                prob_thresh=pt,
                                min_size=ms
                            )
                            p_sum = mask.sum()
                            if p_sum == 0:
                                pos_dices.append(0.0)
                            else:
                                pred_pos_count += 1
                                inter = np.logical_and(mask, c_tgt[idx]).sum()
                                pos_dices.append(float(2.0 * inter / (p_sum + t_sum)))
                                
                    for idx in neg_indices:
                        p_cls = c_cls[idx]
                        if p_cls < ct:
                            neg_fp.append(1.0)
                            neg_emp.append(1.0)
                        else:
                            mask = postprocess_mask(
                                c_seg[idx].astype(np.float32),
                                cls_prob=p_cls,
                                cls_idx=c,
                                cls_thresh=ct,
                                prob_thresh=pt,
                                min_size=ms
                            )
                            p_sum = mask.sum()
                            if p_sum == 0:
                                neg_fp.append(1.0)
                                neg_emp.append(1.0)
                            else:
                                pred_pos_count += 1
                                penalty = min(1.0, float(p_sum) / 100.0)
                                neg_fp.append(max(0.0, 1.0 - penalty))
                                neg_emp.append(0.0)
                                
                    s_dice = float(np.mean(pos_dices)) if pos_dices else 0.0
                    s_fp = float(np.mean(neg_fp)) if neg_fp else 1.0
                    s_emp = float(np.mean(neg_emp)) if neg_emp else 1.0
                    score = 0.70 * s_dice + 0.20 * s_fp + 0.10 * s_emp
                    
                    top_configs.append({
                        'cls': ct, 'prob': pt, 'min_size': ms,
                        'score': score, 's_dice': s_dice, 's_fp': s_fp, 's_emp': s_emp,
                        'pred_pos': pred_pos_count, 'pred_rate': pred_pos_count / N
                    })
                    
        # Sort by score descending
        top_configs.sort(key=lambda x: x['score'], reverse=True)
        best_results[c] = top_configs[0]
        
        print(f"Top 3 configs for {c_name}:")
        for k in range(min(3, len(top_configs))):
            cfg = top_configs[k]
            print(f"  #{k+1}: ct={cfg['cls']:.2f}, pt={cfg['prob']:.2f}, ms={cfg['min_size']:3d} -> Score={cfg['score']:.5f} "
                  f"(Dice={cfg['s_dice']:.4f}, FP={cfg['s_fp']:.4f}, Emp={cfg['s_emp']:.4f}) | "
                  f"Pred={cfg['pred_pos']}/{N} ({cfg['pred_rate']*100:.1f}%)")
                  
    print("\n" + "=" * 80)
    print("BEST OVERALL CONFIGURATION:")
    print("=" * 80)
    total_opt_score = 0
    for c in range(4):
        b = best_results[c]
        print(f"  {DEFECT_TYPES[c]:8s}: ct={b['cls']:.2f}, pt={b['prob']:.2f}, ms={b['min_size']:3d} | Score={b['score']:.5f} | Pred Rate={b['pred_rate']*100:.1f}%")
        total_opt_score += 0.25 * b['score']
    print(f"Projected Validation Composite Score: {total_opt_score:.5f}")

if __name__ == '__main__':
    run_search()
