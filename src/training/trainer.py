"""
Training Module for Gold Price Forecasting
==========================================

Implements a comprehensive training pipeline with:
- Gradient accumulation for effective large batch training
- Mixed precision training support
- MLflow experiment tracking
- Comprehensive logging and monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any, Tuple
from pathlib import Path
import time
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Training parameters
    epochs: int = 200
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    
    # Optimizer parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer_type: str = "adamw"
    
    # Scheduler parameters
    scheduler_type: str = "cosine"
    warmup_epochs: int = 10
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0001
    
    # Regularization
    gradient_clip: float = 1.0
    weight_constraint: float = 3.0
    
    # Mixed precision
    use_mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    
    # Device
    device: str = "auto"
    
    # Seeds
    seed: int = 42


class Trainer:
    """
    Comprehensive trainer for deep learning models.
    
    Features:
    - Gradient accumulation for effective batch size
    - Early stopping with patience
    - Model checkpointing
    - MLflow experiment tracking
    - Training history logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        config: TrainingConfig,
        mlflow_tracking: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            config: Training configuration
            mlflow_tracking: Whether to use MLflow tracking
        """
        self.config = config
        self.mlflow_tracking = mlflow_tracking
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.history: Dict[str, List] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'train_loss_components': [],
            'val_metrics': []
        }
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # MLflow setup
        if mlflow_tracking:
            self._setup_mlflow()
    
    def _setup_optimizer(self):
        """Setup optimizer with weight decay."""
        from .optimizers import create_optimizer
        self.optimizer = create_optimizer(
            self.model,
            self.config.optimizer_type,
            self.config.learning_rate,
            self.config.weight_decay
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        from .optimizers import create_scheduler
        
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.scheduler_type,
            total_steps,
            warmup_steps
        )
    
    def _setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        try:
            import mlflow
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("gold_price_forecasting")
            self.mlflow = mlflow
            logger.info("MLflow tracking enabled")
        except ImportError:
            logger.warning("MLflow not installed, disabling tracking")
            self.mlflow_tracking = False
            self.mlflow = None
    
    def train(self) -> Dict[str, Any]:
        """
        Run complete training loop.
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        if self.mlflow_tracking:
            self.mlflow.start_run()
            self._log_config_to_mlflow()
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training phase
                train_results = self._train_epoch()
                
                # Validation phase
                val_results = self._validate()
                
                epoch_time = time.time() - epoch_start
                
                # Update history
                self.history['train_loss'].append(train_results['loss'])
                self.history['val_loss'].append(val_results['loss'])
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                self.history['epoch_time'].append(epoch_time)
                self.history['train_loss_components'].append(train_results.get('components', {}))
                self.history['val_metrics'].append(val_results)
                
                # Logging
                self._log_epoch(epoch, train_results, val_results, epoch_time)
                
                # Checkpointing
                is_best = val_results['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_results['loss']
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Final logging
            results = self._compile_results()
            
            if self.mlflow_tracking:
                self._log_final_results(results)
                self.mlflow.end_run()
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.mlflow_tracking:
                self.mlflow.end_run(status="FAILED")
            raise
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        loss_components = {'huber_loss': 0.0, 'directional_loss': 0.0}
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            sequences = batch['sequences'].to(self.device)
            targets = batch['targets'].to(self.device)
            volatility = batch.get('volatility')
            if volatility is not None:
                volatility = volatility.to(self.device).unsqueeze(-1)
            
            # Forward pass
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(sequences, volatility)
                    loss_dict = self._compute_loss(outputs, targets, sequences)
                    loss = loss_dict['loss'] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(sequences, volatility)
                loss_dict = self._compute_loss(outputs, targets, sequences)
                loss = loss_dict['loss'] / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Accumulate losses
            total_loss += loss_dict['loss'].item() * len(sequences)
            total_samples += len(sequences)
            
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key].item() * len(sequences)
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / total_samples,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Normalize loss components
        for key in loss_components:
            loss_components[key] /= total_samples
        
        return {
            'loss': total_loss / total_samples,
            'components': loss_components
        }
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        sequences: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with optional directional component."""
        predictions = outputs['predictions']
        
        # Get previous values for directional loss
        previous_values = sequences[:, -1, 0]  # Last closing price in sequence
        
        if hasattr(self.loss_fn, 'forward') and 'previous_values' in self.loss_fn.forward.__code__.co_varnames:
            loss_dict = self.loss_fn(predictions, targets, previous_values)
        else:
            loss = self.loss_fn(predictions.view(-1), targets.view(-1))
            loss_dict = {'loss': loss}
        
        return loss_dict
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for batch in self.val_loader:
            sequences = batch['sequences'].to(self.device)
            targets = batch['targets'].to(self.device)
            volatility = batch.get('volatility')
            if volatility is not None:
                volatility = volatility.to(self.device).unsqueeze(-1)
            
            outputs = self.model(sequences, volatility)
            loss_dict = self._compute_loss(outputs, targets, sequences)
            
            total_loss += loss_dict['loss'].item() * len(sequences)
            total_samples += len(sequences)
            
            all_predictions.extend(outputs['predictions'].cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
        
        # Compute additional metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        
        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.diff(predictions) > 0
            target_direction = np.diff(targets) > 0
            directional_accuracy = np.mean(pred_direction == target_direction)
        else:
            directional_accuracy = 0.0
        
        return {
            'loss': total_loss / total_samples,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def _log_epoch(
        self,
        epoch: int,
        train_results: Dict,
        val_results: Dict,
        epoch_time: float
    ):
        """Log epoch results."""
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_results['loss']:.6f}, "
            f"Val Loss: {val_results['loss']:.6f}, "
            f"Val MAE: {val_results['mae']:.4f}, "
            f"Val RMSE: {val_results['rmse']:.4f}, "
            f"Dir Acc: {val_results['directional_accuracy']:.4f}, "
            f"Time: {epoch_time:.1f}s"
        )
        
        if self.mlflow_tracking:
            self.mlflow.log_metrics({
                'train_loss': train_results['loss'],
                'val_loss': val_results['loss'],
                'val_mae': val_results['mae'],
                'val_rmse': val_results['rmse'],
                'val_directional_accuracy': val_results['directional_accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }, step=epoch)
    
    def _log_config_to_mlflow(self):
        """Log configuration to MLflow."""
        config_dict = {
            'epochs': self.config.epochs,
            'batch_size': self.config.batch_size,
            'effective_batch_size': self.config.batch_size * self.config.gradient_accumulation_steps,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'optimizer': self.config.optimizer_type,
            'scheduler': self.config.scheduler_type,
            'early_stopping_patience': self.config.early_stopping_patience,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        self.mlflow.log_params(config_dict)
    
    def _log_final_results(self, results: Dict):
        """Log final results to MLflow."""
        self.mlflow.log_metrics({
            'best_val_loss': results['best_val_loss'],
            'final_train_loss': results['final_train_loss'],
            'total_epochs': results['total_epochs'],
            'training_time': results['training_time']
        })
        
        # Log model
        self.mlflow.pytorch.log_model(self.model, "model")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.__dict__
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved best model to {path}")
        
        if not self.config.save_best_only:
            path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_epoch = checkpoint['epoch']
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from {path}")
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile training results."""
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'total_epochs': self.current_epoch + 1,
            'training_time': sum(self.history['epoch_time']),
            'history': self.history
        }
