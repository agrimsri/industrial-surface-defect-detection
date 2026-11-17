import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for binary segmentation.
    Computes soft dice over batch and spatial dimensions.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        # probs, targets: (B, C, H, W)
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        intersection = (probs * targets).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice.mean()
        return loss

class JointMultiTaskLoss(nn.Module):
    """
    Joint loss function optimizing both:
    1. Segmentation: BCEWithLogits + SoftDiceLoss
    2. Classification: BCEWithLogits on image-level defect presence
    Supports positive class weighting (pos_weight) for minority class boosting.
    """
    def __init__(self, cls_weight=0.5, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.cls_weight = cls_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1))
        else:
            self.bce_cls = nn.BCEWithLogitsLoss()
            self.bce_seg = nn.BCEWithLogitsLoss()
            
        self.dice_seg = SoftDiceLoss()

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        # 1. Classification loss
        l_cls = self.bce_cls(cls_logits, cls_targets)
        
        # 2. Segmentation losses
        l_bce = self.bce_seg(seg_logits, seg_targets)
        probs = torch.sigmoid(seg_logits)
        l_dice = self.dice_seg(probs, seg_targets)
        
        l_seg = self.bce_weight * l_bce + self.dice_weight * l_dice
        l_total = l_seg + self.cls_weight * l_cls
        
        return {
            'total_loss': l_total,
            'seg_loss': l_seg,
            'cls_loss': l_cls,
            'bce_loss': l_bce,
            'dice_loss': l_dice
        }
