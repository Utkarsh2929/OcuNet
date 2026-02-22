#!/usr/bin/env python3
"""
Improved Multi-Label Training with Warmup, Cosine Annealing, and EMA.
Fixed scheduler initialization issue.
"""

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
import math
import warnings

# Suppress the scheduler warning (it's harmless but annoying)
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*')


class WarmupCosineScheduler:
    """
    Custom scheduler with linear warmup and cosine decay.
    Avoids PyTorch's LR scheduler initialization issues.
    """

    def __init__(
            self,
            optimizer,
            warmup_steps: int,
            total_steps: int,
            min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (self.current_step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self):
        """Update learning rate."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class MultiLabelMetrics:
    """Compute multi-label metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_probs = []
        self.all_targets = []

    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        self.all_probs.append(probs.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        if not self.all_probs:
            return {'f1_macro': 0, 'f1_micro': 0, 'precision_macro': 0,
                    'recall_macro': 0, 'mAP': 0, 'roc_auc': 0, 'hamming_loss': 0}

        probs = np.vstack(self.all_probs)
        targets = np.vstack(self.all_targets)
        preds = (probs >= self.threshold).astype(np.float32)

        metrics = {}

        try:
            metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
            metrics['f1_micro'] = f1_score(targets, preds, average='micro', zero_division=0)
            metrics['precision_macro'] = precision_score(targets, preds, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(targets, preds, average='macro', zero_division=0)
        except Exception:
            metrics['f1_macro'] = 0.0
            metrics['f1_micro'] = 0.0
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0

        try:
            metrics['mAP'] = average_precision_score(targets, probs, average='macro')
        except Exception:
            metrics['mAP'] = 0.0

        try:
            valid_classes = []
            for i in range(targets.shape[1]):
                if 0 < targets[:, i].sum() < len(targets):
                    valid_classes.append(i)
            if valid_classes:
                metrics['roc_auc'] = roc_auc_score(
                    targets[:, valid_classes], probs[:, valid_classes], average='macro'
                )
            else:
                metrics['roc_auc'] = 0.0
        except Exception:
            metrics['roc_auc'] = 0.0

        metrics['hamming_loss'] = (preds != targets).mean()

        return metrics


class EarlyStopping:
    """Early stopping."""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ImprovedTrainer:
    """Improved trainer with warmup, EMA, and better scheduling."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            config: dict,
            device: torch.device,
            class_names: List[str]
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.class_names = class_names

        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.threshold = config['training'].get('threshold', 0.5)
        self.warmup_epochs = config['training'].get('warmup_epochs', 5)

        # Gradient accumulation for larger effective batch sizes
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        effective_batch = config['training']['batch_size'] * self.grad_accum_steps
        if self.grad_accum_steps > 1:
            print(f"  Gradient accumulation: {self.grad_accum_steps} steps (effective batch={effective_batch})")

        # Hard negative mining
        self.use_hard_negatives = config['training'].get('hard_negative_mining', False)
        self.hard_neg_weight = config['training'].get('hard_negative_weight', 3.0)
        self.sample_weights_boost = None  # Updated after each epoch

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config['training']['weight_decay']
        )

        # Calculate steps
        steps_per_epoch = len(train_loader)
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps = self.num_epochs * steps_per_epoch

        # Custom scheduler (no warnings!)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-7
        )

        # EMA
        self.ema = EMA(self.model, decay=0.999)

        # Mixed precision
        self.use_amp = device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience']
        )

        # Checkpointing
        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.metrics = MultiLabelMetrics(self.threshold)

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_mAP': [], 'val_mAP': [],
            'learning_rate': []
        }

        self.best_val_f1 = 0.0
        self.best_val_mAP = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[float, Dict]:
        """Train one epoch with gradient accumulation."""
        self.model.train()
        self.metrics.reset()

        running_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.ema.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.ema.update()

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                self.metrics.update(probs, labels)

            running_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item() * self.grad_accum_steps:.4f}', 'lr': f'{current_lr:.2e}'})

        return running_loss / num_batches, self.metrics.compute()

    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> Tuple[float, Dict]:
        """Validate with optional EMA weights."""
        if use_ema:
            self.ema.apply_shadow()

        self.model.eval()
        self.metrics.reset()

        running_loss = 0.0
        num_batches = 0

        for images, labels in tqdm(self.val_loader, desc="Validating", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            self.metrics.update(probs, labels)

            running_loss += loss.item()
            num_batches += 1

        if use_ema:
            self.ema.restore()

        return running_loss / num_batches, self.metrics.compute()

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        # Use EMA weights for saving
        self.ema.apply_shadow()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_f1': self.best_val_f1,
            'best_val_mAP': self.best_val_mAP,
            'config': self.config,
            'class_names': self.class_names,
            'history': self.history,
            'threshold': self.threshold
        }

        self.ema.restore()

        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pth')

        if is_best:
            # Save with EMA weights
            self.ema.apply_shadow()
            checkpoint['model_state_dict'] = self.model.state_dict()
            self.ema.restore()

            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"  [SAVED] Best model - F1: {metrics['f1_macro']:.4f}, mAP: {metrics['mAP']:.4f}")

    def train(self) -> Dict:
        """Full training loop."""
        print("=" * 70)
        print("IMPROVED MULTI-LABEL TRAINING")
        print("=" * 70)
        print(f"Model: {self.config['model']['architecture']}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Warmup: {self.warmup_epochs} epochs")
        print(f"Classes: {len(self.class_names)}")
        print(f"Batches/epoch: {len(self.train_loader)}")
        print(f"Learning Rate: {self.scheduler.min_lr:.2e} -> {self.learning_rate:.6f} (peak)")
        print(f"EMA: Enabled (decay=0.999)")
        print("=" * 70)

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate with EMA
            val_loss, val_metrics = self.validate(use_ema=True)

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['train_mAP'].append(train_metrics['mAP'])
            self.history['val_mAP'].append(val_metrics['mAP'])
            self.history['learning_rate'].append(current_lr)

            # Print
            print(
                f"  Train - Loss: {train_loss:.4f} | F1: {train_metrics['f1_macro']:.4f} | mAP: {train_metrics['mAP']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | F1: {val_metrics['f1_macro']:.4f} | mAP: {val_metrics['mAP']:.4f}")
            print(
                f"  LR: {current_lr:.2e} | ROC-AUC: {val_metrics['roc_auc']:.4f} | Recall: {val_metrics['recall_macro']:.4f}")

            # Check best
            is_best = val_metrics['f1_macro'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_val_mAP = val_metrics['mAP']
                self.best_epoch = epoch

            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.early_stopping(val_metrics['f1_macro']):
                print(
                    f"\n[EARLY STOPPING] No improvement for {self.early_stopping.patience} epochs. Stopping at epoch {epoch}.")
                break

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Best Validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"Best Validation mAP: {self.best_val_mAP:.4f}")
        print("=" * 70)

        return self.history