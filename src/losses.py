import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """
    Differentiable Foreground Soft Dice Loss for binary segmentation.
    Only computes Dice loss over instances where foreground defect is present,
    matching Kaggle's foreground-only Dice evaluation specification.
    Prevents artificial 1.0 Dice penalties on empty slices from driving
    minority class (defect_a, defect_b) sigmoid probabilities to 0.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # probs, targets: (B, C, H, W)
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        t_sum = targets.sum(dim=2)
        p_sum = probs.sum(dim=2)
        intersection = (probs * targets).sum(dim=2)
        
        pos_mask = (t_sum > 0).float()
        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)
        
        num_pos = pos_mask.sum()
        if num_pos > 0:
            loss = (pos_mask * (1.0 - dice)).sum() / num_pos
        else:
            # Differentiable graph-connected zero loss
            loss = (probs * 0.0).sum()
        return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension with respect to sorted errors.
    See Berman et al., CVPR 2018.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, targets):
    """
    Binary Lovasz hinge loss on 1D flattened tensors.
    logits: [P] Variable, logits at each pixel prediction
    targets: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(logits) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * targets.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = targets[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

class LovaszHingeLoss(nn.Module):
    """
    Lovász-Hinge loss for binary segmentation fine-tuning.
    Optimizes the Jaccard index / IoU surrogate directly via Lovász extension of submodular losses.
    When only_present=True, only computes loss over foreground-positive instances (targets.sum() > 0),
    matching competition metric dynamics and preventing artificial penalties on empty slices.
    """
    def __init__(self, per_image=True, only_present=True):
        super().__init__()
        self.per_image = per_image
        self.only_present = only_present

    def forward(self, logits, targets):
        if self.per_image:
            B, C, H, W = logits.shape
            total_loss = 0.0
            count = 0
            for b in range(B):
                for c in range(C):
                    t = targets[b, c]
                    is_pos = (t.sum() > 0)
                    if self.only_present and not is_pos:
                        continue
                    l = logits[b, c].reshape(-1)
                    t_flat = t.reshape(-1)
                    total_loss = total_loss + lovasz_hinge_flat(l, t_flat)
                    count += 1
            if count > 0:
                return total_loss / count
            else:
                return (logits * 0.0).sum()
        else:
            if self.only_present:
                B, C, H, W = logits.shape
                pos_mask = (targets.view(B, C, -1).sum(dim=2) > 0)
                if not pos_mask.any():
                    return (logits * 0.0).sum()
                logits_pos = logits.permute(0, 1, 2, 3)[pos_mask].view(-1)
                targets_pos = targets.permute(0, 1, 2, 3)[pos_mask].view(-1)
                return lovasz_hinge_flat(logits_pos, targets_pos)
            else:
                return lovasz_hinge_flat(logits.view(-1), targets.view(-1))

class JointMultiTaskLoss(nn.Module):
    """
    Joint loss function optimizing both:
    1. Segmentation: BCEWithLogits + SoftDiceLoss (+ optional LovaszHingeLoss)
    2. Classification: BCEWithLogits on image-level defect presence
    Supports positive class weighting (pos_weight) for minority class boosting.
    """
    def __init__(self, cls_weight=0.5, bce_weight=0.5, dice_weight=0.5, lovasz_weight=0.0, pos_weight=None):
        super().__init__()
        self.cls_weight = cls_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1))
        else:
            self.bce_cls = nn.BCEWithLogitsLoss()
            self.bce_seg = nn.BCEWithLogitsLoss()
            
        self.dice_seg = SoftDiceLoss()
        self.lovasz_seg = LovaszHingeLoss(per_image=True, only_present=True)

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        # 1. Classification loss
        l_cls = self.bce_cls(cls_logits, cls_targets)
        
        # 2. Segmentation losses
        l_bce = self.bce_seg(seg_logits, seg_targets)
        probs = torch.sigmoid(seg_logits)
        l_dice = self.dice_seg(probs, seg_targets)
        
        l_seg = self.bce_weight * l_bce + self.dice_weight * l_dice
        
        if self.lovasz_weight > 0.0:
            l_lovasz = self.lovasz_seg(seg_logits, seg_targets)
            l_seg = l_seg + self.lovasz_weight * l_lovasz
        else:
            l_lovasz = torch.tensor(0.0, device=cls_logits.device)
            
        l_total = l_seg + self.cls_weight * l_cls
        
        return {
            'total_loss': l_total,
            'seg_loss': l_seg,
            'cls_loss': l_cls,
            'bce_loss': l_bce,
            'dice_loss': l_dice,
            'lovasz_loss': l_lovasz
        }
