"""
Training Callbacks
==================

Implements training callbacks for monitoring and control:
- Early stopping
- Model checkpointing
- Learning rate scheduling
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0001,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore model to best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_weights: Optional[Dict] = None
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y - min_delta
        else:
            self.is_better = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered. Best score: {self.best_score:.6f}")
        
        return self.should_stop
    
    def restore_weights(self, model: nn.Module):
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best model weights")


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Whether to save only best model
            save_freq: Save frequency in epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_score: Optional[float] = None
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y
        else:
            self.is_better = lambda x, y: x > y
    
    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float]
    ):
        """
        Save checkpoint if conditions are met.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            metrics: Current metrics
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return
        
        # Check if we should save
        should_save = False
        
        if self.best_score is None or self.is_better(current_score, self.best_score):
            self.best_score = current_score
            should_save = True
            is_best = True
        else:
            is_best = False
            should_save = not self.save_best_only and (epoch + 1) % self.save_freq == 0
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_score': self.best_score
            }
            
            if is_best:
                path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint, path)
                logger.info(f"Saved best model (score: {current_score:.6f}) to {path}")
            else:
                path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, path)
                logger.info(f"Saved checkpoint to {path}")


class LRSchedulerCallback:
    """
    Learning rate scheduler callback with warmup support.
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        step_on: str = 'epoch'
    ):
        """
        Initialize LR scheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler
            step_on: When to step ('epoch' or 'batch')
        """
        self.scheduler = scheduler
        self.step_on = step_on
    
    def on_batch_end(self):
        """Called at end of each batch."""
        if self.step_on == 'batch':
            self.scheduler.step()
    
    def on_epoch_end(self, metrics: Optional[Dict] = None):
        """Called at end of each epoch."""
        if self.step_on == 'epoch':
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None and 'val_loss' in metrics:
                    self.scheduler.step(metrics['val_loss'])
            else:
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]


class GradientMonitor:
    """
    Monitor gradient statistics during training.
    """
    
    def __init__(self, model: nn.Module, log_freq: int = 100):
        """
        Initialize gradient monitor.
        
        Args:
            model: Model to monitor
            log_freq: Frequency of logging (in batches)
        """
        self.model = model
        self.log_freq = log_freq
        self.step_count = 0
        
        self.gradient_history: Dict[str, list] = {
            'mean': [],
            'std': [],
            'max': [],
            'min': []
        }
    
    def __call__(self):
        """Compute and log gradient statistics."""
        self.step_count += 1
        
        if self.step_count % self.log_freq != 0:
            return
        
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.cpu().numpy().flatten())
        
        if grads:
            all_grads = np.concatenate(grads)
            stats = {
                'mean': np.mean(np.abs(all_grads)),
                'std': np.std(all_grads),
                'max': np.max(np.abs(all_grads)),
                'min': np.min(np.abs(all_grads))
            }
            
            for key, value in stats.items():
                self.gradient_history[key].append(value)
            
            logger.debug(
                f"Gradient stats - Mean: {stats['mean']:.6f}, "
                f"Std: {stats['std']:.6f}, Max: {stats['max']:.6f}"
            )
    
    def get_history(self) -> Dict[str, list]:
        """Get gradient history."""
        return self.gradient_history
