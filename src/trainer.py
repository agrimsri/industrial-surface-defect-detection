import os
import gc
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DefectDataset, DEFECT_TYPES
from src.losses import JointMultiTaskLoss
from src.metrics import StreamingMetricMeter, compute_competition_metric
from src.postprocess import postprocess_mask

class Trainer:
    """
    Production-Grade Trainer with:
    - PyTorch Mixed Precision (AMP FP16)
    - Gradient Scaler & Gradient Clipping
    - Post-epoch weight & gradient inspection (learning dynamics analysis)
    - Streaming per-batch metric evaluation (OOM-proof)
    - Inter-epoch garbage collection and GPU cache clearing
    - Early stopping on composite competition metric
    - Fail-fast divergence protection
    - Seamless checkpoint resumption
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        criterion=None,
        device='cuda',
        checkpoint_dir='checkpoints',
        experiment_name='exp_run',
        patience=3,
        min_delta=1e-4,
        resume_checkpoint=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        
        # Enforce memory hygiene on validation loader: num_workers=0, pin_memory=False
        if val_loader is not None and (getattr(val_loader, 'pin_memory', False) or getattr(val_loader, 'num_workers', 0) > 0):
            self.val_loader = DataLoader(
                val_loader.dataset,
                batch_size=val_loader.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=val_loader.collate_fn,
                drop_last=val_loader.drop_last
            )
        else:
            self.val_loader = val_loader
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else JointMultiTaskLoss()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.patience = patience
        self.min_delta = min_delta
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        is_cuda = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_final_score': [],
            'val_dice': [],
            'val_fp_score': [],
            'val_empty_acc': [],
            'grad_norm': [],
            'weight_stats': []
        }
        self.best_score = -float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.start_epoch = 1
        
        if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: could not restore optimizer state: {e}")
            self.best_score = ckpt.get('best_score', -float('inf'))
            self.best_epoch = ckpt.get('epoch', 0)
            self.start_epoch = self.best_epoch + 1
            print(f"Resumed from checkpoint {resume_checkpoint} at epoch {self.best_epoch} (Best score: {self.best_score:.5f})")

    def analyze_weights_and_gradients(self):
        """
        Inspects layer weights and gradients to verify healthy numerical dynamics.
        """
        total_norm = 0.0
        param_stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                w_mean = param.data.mean().item()
                w_std = param.data.std().item()
                w_max = param.data.abs().max().item()
                if 'conv' in name or 'cls_head' in name or 'seg_head' in name:
                    param_stats[name] = {'mean': w_mean, 'std': w_std, 'max': w_max}
                    
        total_norm = math.sqrt(total_norm)
        return total_norm, param_stats

    def train_epoch(self, epoch):
        self.model.train()
        running_losses = {'total': 0.0, 'seg': 0.0, 'cls': 0.0}
        total_steps = len(self.train_loader)
        
        t0 = time.time()
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            cls_targets = batch['cls_label'].to(self.device, non_blocking=True)
            seg_targets = batch['mask'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda'), dtype=torch.float16):
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    cls_logits, seg_logits = outputs
                else:
                    seg_logits = outputs
                    cls_logits = torch.amax(seg_logits, dim=(2, 3))
                    
                loss_dict = self.criterion(cls_logits, seg_logits, cls_targets, seg_targets)
                loss = loss_dict['total_loss']
                
            # Fail-fast divergence check
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Divergence detected at epoch {epoch}, step {step}! Loss is {loss.item()}")
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_losses['total'] += loss.item()
            running_losses['seg'] += loss_dict['seg_loss'].item()
            running_losses['cls'] += loss_dict['cls_loss'].item()
            
            del images, cls_targets, seg_targets, outputs, loss_dict, loss, batch
            
        avg_losses = {k: v / total_steps for k, v in running_losses.items()}
        epoch_grad_norm, weight_stats = self.analyze_weights_and_gradients()
        epoch_time = time.time() - t0
        
        # Memory hygiene post-train epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return avg_losses, epoch_grad_norm, weight_stats, epoch_time

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        total_steps = len(self.val_loader)
        
        num_classes = getattr(self.model, 'num_classes', 4)
        meter = StreamingMetricMeter(num_classes=num_classes)
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device, non_blocking=True)
            cls_targets = batch['cls_label'].to(self.device, non_blocking=True)
            seg_targets = batch['mask'].to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda'), dtype=torch.float16):
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    cls_logits, seg_logits = outputs
                else:
                    seg_logits = outputs
                    cls_logits = torch.amax(seg_logits, dim=(2, 3))
                    
                loss_dict = self.criterion(cls_logits, seg_logits, cls_targets, seg_targets)
                running_loss += loss_dict['total_loss'].item()
                
            cls_probs = torch.sigmoid(cls_logits).cpu().numpy()  # (B, 4)
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()  # (B, 4, 128, 800)
            targets_np = (seg_targets.cpu().numpy() > 0).astype(np.uint8)  # (B, 4, 128, 800)
            
            # Post-process batch predictions using classification gating + min size
            B, C, H, W = seg_probs.shape
            batch_binary_preds = np.zeros((B, C, H, W), dtype=np.uint8)
            for b in range(B):
                for c in range(C):
                    batch_binary_preds[b, c] = postprocess_mask(
                        seg_probs[b, c],
                        cls_prob=cls_probs[b, c],
                        cls_idx=c
                    )
                    
            # Accumulate evaluation metrics in a streaming fashion per-batch
            meter.update(batch_binary_preds, targets_np)
            
            del images, cls_targets, seg_targets, outputs, cls_logits, seg_logits
            del cls_probs, seg_probs, targets_np, batch_binary_preds, loss_dict, batch
            
        metrics = meter.compute()
        metrics['val_loss'] = running_loss / total_steps if total_steps > 0 else 0.0
        
        # Memory hygiene post-validation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return metrics

    def fit(self, epochs):
        print(f"=== Starting Training [{self.experiment_name}] on {self.device} ===")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        if self.start_epoch > epochs:
            print(f"Already trained up to epoch {self.start_epoch - 1} (target: {epochs}). Skipping fit.")
            return self.history
            
        for epoch in range(self.start_epoch, epochs + 1):
            train_losses, grad_norm, weight_stats, epoch_time = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            current_score = val_metrics['final_score']
            
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_final_score'].append(current_score)
            self.history['val_dice'].append(val_metrics['s_dice'])
            self.history['val_fp_score'].append(val_metrics['s_fp'])
            self.history['val_empty_acc'].append(val_metrics['s_empty'])
            self.history['grad_norm'].append(grad_norm)
            
            # Epoch Log
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
                f"Train Loss: {train_losses['total']:.4f} (Seg: {train_losses['seg']:.4f}, Cls: {train_losses['cls']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Score: {current_score:.5f} (Dice: {val_metrics['s_dice']:.4f}, FP: {val_metrics['s_fp']:.4f}, Emp: {val_metrics['s_empty']:.4f}) | "
                f"GradNorm: {grad_norm:.2f}"
            )
            
            # Checkpoint & Early Stopping Check
            if current_score > self.best_score + self.min_delta:
                improvement = current_score - self.best_score
                self.best_score = current_score
                self.best_epoch = epoch
                self.patience_counter = 0
                
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_score': self.best_score,
                    'val_metrics': val_metrics,
                }, ckpt_path)
                print(f"  --> Saved new best checkpoint to {ckpt_path} (+{improvement:.5f})")
            else:
                self.patience_counter += 1
                print(f"  --> No score improvement ({self.patience_counter}/{self.patience})")
                if self.patience_counter >= self.patience:
                    print(f"=== Early Stopping triggered at epoch {epoch}! Best score: {self.best_score:.5f} at epoch {self.best_epoch} ===")
                    break
                    
            # Memory hygiene between epochs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        print(f"=== Training Completed. Best Val Score: {self.best_score:.5f} (Epoch {self.best_epoch}) ===")
        return self.history
