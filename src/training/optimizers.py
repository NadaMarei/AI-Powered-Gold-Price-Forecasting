"""
Optimizer and Scheduler Utilities
=================================

Factory functions for creating optimizers and learning rate schedulers
with proper configuration for financial time series forecasting.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LambdaLR
)
from typing import Optional, Dict, Any, List
import math
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with proper parameter groups.
    
    Separates parameters into those that should have weight decay
    and those that shouldn't (biases, layer norms).
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and layer norm parameters
        if 'bias' in name or 'layer_norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type == 'adam':
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(
            param_groups,
            lr=learning_rate,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(
        f"Created {optimizer_type} optimizer with lr={learning_rate}, "
        f"weight_decay={weight_decay}, "
        f"decay_params={len(decay_params)}, no_decay_params={len(no_decay_params)}"
    )
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 10000,
    warmup_steps: int = 1000,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler with optional warmup.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler or None
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'none':
        return None
    
    if scheduler_type == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'cosine_restarts':
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', total_steps // 4),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    
    elif scheduler_type == 'linear':
        # Linear decay with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', total_steps // 3),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'multistep':
        milestones = kwargs.get('milestones', [total_steps // 3, 2 * total_steps // 3])
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(
        f"Created {scheduler_type} scheduler with "
        f"total_steps={total_steps}, warmup_steps={warmup_steps}"
    )
    
    return scheduler


class WarmupScheduler:
    """
    Wrapper for warmup + base scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            base_scheduler: Base scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            warmup_factor = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor
        else:
            self.base_scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rate."""
        return [pg['lr'] for pg in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict:
        """Get scheduler state."""
        return {
            'current_step': self.current_step,
            'base_scheduler': self.base_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
