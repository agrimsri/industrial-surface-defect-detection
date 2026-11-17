import numpy as np
import torch

def compute_instance_dice(pred_mask, true_mask, eps=1e-7):
    """
    Computes Dice coefficient for a single binary mask pair.
    """
    p = (pred_mask > 0).astype(np.float32)
    t = (true_mask > 0).astype(np.float32)
    p_sum = p.sum()
    t_sum = t.sum()
    
    # Ground truth is empty
    if t_sum == 0:
        return 1.0 if p_sum == 0 else 0.0
        
    # Ground truth is positive
    if p_sum == 0:
        return 0.0
        
    intersection = (p * t).sum()
    return float((2.0 * intersection) / (p_sum + t_sum))


class StreamingMetricMeter:
    """
    Streaming accumulator for the Kaggle competition composite metric:
      Final Score = 0.70 * S_dice + 0.20 * S_fp + 0.10 * S_empty
      
    Accumulates per-instance scalar metrics (Dice for positive instances,
    false-positive penalty and empty accuracy for clean surfaces) batch-by-batch
    in a memory-efficient manner without storing massive 4D mask tensors in RAM.
    """
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.pos_dices = []
        self.class_pos_dices = {c: [] for c in range(self.num_classes)}
        self.neg_fp_scores = []
        self.neg_empty_correct = []
        self.class_neg_fp = {c: [] for c in range(self.num_classes)}
        self.class_neg_empty = {c: [] for c in range(self.num_classes)}
        
    def update(self, preds, targets):
        """
        Updates the metric accumulator with a batch of predictions and targets.
        
        Args:
          preds: np.ndarray or torch.Tensor of shape (B, C, H, W)
                 binary predictions (0 or 1, or boolean)
          targets: np.ndarray or torch.Tensor of shape (B, C, H, W)
                   ground truth binary masks (0 or 1, or boolean)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        B, C, H, W = targets.shape
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C}"
        
        # Flatten spatial dimensions for fast summation
        p_flat = (preds > 0).reshape(B, C, -1)
        t_flat = (targets > 0).reshape(B, C, -1)
        
        p_sums = p_flat.sum(axis=-1, dtype=np.int64)
        t_sums = t_flat.sum(axis=-1, dtype=np.int64)
        
        for b in range(B):
            for c in range(C):
                t_sum = int(t_sums[b, c])
                p_sum = int(p_sums[b, c])
                
                if t_sum > 0:
                    # Positive ground truth instance: compute Dice
                    if p_sum == 0:
                        dice = 0.0
                    else:
                        inter = int(np.logical_and(p_flat[b, c], t_flat[b, c]).sum())
                        dice = float((2.0 * inter) / (p_sum + t_sum))
                    self.pos_dices.append(dice)
                    self.class_pos_dices[c].append(dice)
                else:
                    # Negative ground truth instance: clean surface
                    if p_sum == 0:
                        fp_score = 1.0
                        empty_score = 1.0
                    else:
                        fp_penalty = min(1.0, float(p_sum) / 500.0)
                        fp_score = max(0.0, 1.0 - fp_penalty)
                        empty_score = 0.0
                    self.neg_fp_scores.append(fp_score)
                    self.neg_empty_correct.append(empty_score)
                    self.class_neg_fp[c].append(fp_score)
                    self.class_neg_empty[c].append(empty_score)
                        
    def compute(self):
        """
        Computes final competition metrics from accumulated scalar scores.
        """
        s_dice = float(np.mean(self.pos_dices)) if len(self.pos_dices) > 0 else 0.0
        s_fp = float(np.mean(self.neg_fp_scores)) if len(self.neg_fp_scores) > 0 else 1.0
        s_empty = float(np.mean(self.neg_empty_correct)) if len(self.neg_empty_correct) > 0 else 1.0
        
        final_score = 0.70 * s_dice + 0.20 * s_fp + 0.10 * s_empty
        
        per_class_pos = {
            c: (float(np.mean(self.class_pos_dices[c])) if len(self.class_pos_dices[c]) > 0 else 0.0)
            for c in range(self.num_classes)
        }
        per_class_fp = {
            c: (float(np.mean(self.class_neg_fp[c])) if len(self.class_neg_fp[c]) > 0 else 1.0)
            for c in range(self.num_classes)
        }
        per_class_empty = {
            c: (float(np.mean(self.class_neg_empty[c])) if len(self.class_neg_empty[c]) > 0 else 1.0)
            for c in range(self.num_classes)
        }
        per_class_score = {
            c: 0.70 * per_class_pos[c] + 0.20 * per_class_fp[c] + 0.10 * per_class_empty[c]
            for c in range(self.num_classes)
        }
        
        return {
            'final_score': final_score,
            's_dice': s_dice,
            's_fp': s_fp,
            's_empty': s_empty,
            'per_class_dice': per_class_pos,
            'per_class_fp': per_class_fp,
            'per_class_empty': per_class_empty,
            'per_class_score': per_class_score,
            'n_pos': len(self.pos_dices),
            'n_neg': len(self.neg_fp_scores)
        }


def compute_competition_metric(preds, targets):
    """
    Computes the official Kaggle competition composite metric:
      Final Score = 0.70 * S_dice + 0.20 * S_fp + 0.10 * S_empty
      
    Args:
      preds: np.ndarray or torch.Tensor (N, 4, 128, 800) binary predictions (0 or 1)
      targets: np.ndarray or torch.Tensor (N, 4, 128, 800) ground truth binary masks
      
    Returns:
      dict with final_score, s_dice, s_fp, s_empty, and per_class_dice
    """
    meter = StreamingMetricMeter(num_classes=targets.shape[1])
    meter.update(preds, targets)
    return meter.compute()
