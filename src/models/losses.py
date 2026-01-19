"""
Custom Loss Functions for Gold Price Forecasting
=================================================

Implements specialized loss functions for financial time series:
1. Huber-Directional Loss - Combines Huber loss with directional accuracy
2. Directional Accuracy Loss - Penalizes incorrect direction predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict


class HuberDirectionalLoss(nn.Module):
    """
    Combined Huber loss with directional accuracy penalty.
    
    This loss function addresses two key aspects of financial forecasting:
    1. Huber loss provides robustness to outliers (extreme price movements)
    2. Directional penalty ensures the model learns to predict price direction
    
    Mathematical Formulation:
        L_total = L_huber + λ * L_directional
        
        L_huber(y, ŷ) = { 0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ
                        { δ * |y - ŷ| - 0.5 * δ²   otherwise
        
        L_directional = mean(max(0, -sign(Δy) * sign(Δŷ)))
        
        where:
            Δy = y_t - y_{t-1}  (actual direction)
            Δŷ = ŷ_t - y_{t-1}  (predicted direction)
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        directional_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize Huber-Directional loss.
        
        Args:
            delta: Huber loss threshold
            directional_weight: Weight for directional component (λ)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.delta = delta
        self.directional_weight = directional_weight
        self.reduction = reduction
        
        self.huber = nn.HuberLoss(delta=delta, reduction=reduction)
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        previous_values: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions [batch, 1] or [batch]
            targets: True values [batch, 1] or [batch]
            previous_values: Previous time step values for direction [batch, 1]
            
        Returns:
            Dictionary with total loss and components
        """
        # Ensure correct shapes
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Huber loss component
        huber_loss = self.huber(predictions, targets)
        
        # Directional loss component
        if previous_values is not None and self.directional_weight > 0:
            previous_values = previous_values.view(-1)
            
            # Compute actual and predicted directions
            actual_direction = torch.sign(targets - previous_values)
            predicted_direction = torch.sign(predictions - previous_values)
            
            # Penalize incorrect directions
            # -1 when directions disagree, +1 when they agree
            direction_agreement = actual_direction * predicted_direction
            
            # Convert to loss (0 for correct, 1 for incorrect)
            directional_loss = F.relu(-direction_agreement + 1) / 2
            
            if self.reduction == 'mean':
                directional_loss = directional_loss.mean()
            elif self.reduction == 'sum':
                directional_loss = directional_loss.sum()
        else:
            directional_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = huber_loss + self.directional_weight * directional_loss
        
        return {
            'loss': total_loss,
            'huber_loss': huber_loss,
            'directional_loss': directional_loss
        }


class DirectionalAccuracyLoss(nn.Module):
    """
    Pure directional accuracy loss.
    
    Useful as an auxiliary loss to encourage correct direction prediction.
    Uses a smooth approximation to make the loss differentiable.
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        """
        Initialize directional accuracy loss.
        
        Args:
            temperature: Temperature for soft sign approximation
            reduction: Reduction method
        """
        super().__init__()
        
        self.temperature = temperature
        self.reduction = reduction
    
    def _soft_sign(self, x: Tensor) -> Tensor:
        """Differentiable approximation of sign function."""
        return torch.tanh(x / self.temperature)
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        previous_values: Tensor
    ) -> Tensor:
        """
        Compute directional loss.
        
        Args:
            predictions: Model predictions [batch]
            targets: True values [batch]
            previous_values: Previous time step values [batch]
            
        Returns:
            Directional loss tensor
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        previous_values = previous_values.view(-1)
        
        # Soft directions
        actual_direction = self._soft_sign(targets - previous_values)
        predicted_direction = self._soft_sign(predictions - previous_values)
        
        # Loss is 1 - agreement (-1 to 1 becomes 0 to 2, then scale to 0 to 1)
        agreement = actual_direction * predicted_direction
        loss = (1 - agreement) / 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedMSELoss(nn.Module):
    """
    MSE loss with sample weights based on volatility regime.
    
    Gives higher weight to samples during high-volatility periods
    where accurate predictions are more valuable.
    """
    
    def __init__(self, base_weight: float = 1.0, volatility_factor: float = 0.5):
        """
        Initialize weighted MSE loss.
        
        Args:
            base_weight: Base weight for all samples
            volatility_factor: Additional weight factor for volatility
        """
        super().__init__()
        
        self.base_weight = base_weight
        self.volatility_factor = volatility_factor
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        volatility: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predictions: Model predictions
            targets: True values
            volatility: Normalized volatility values for weighting
            
        Returns:
            Weighted MSE loss
        """
        mse = (predictions - targets) ** 2
        
        if volatility is not None:
            # Higher volatility = higher weight
            weights = self.base_weight + self.volatility_factor * volatility.view(-1)
            weighted_mse = (weights * mse.view(-1)).mean()
            return weighted_mse
        
        return mse.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes over/under predictions differently.
    
    Useful when the cost of under-predicting differs from over-predicting
    (e.g., in risk-averse trading strategies).
    """
    
    def __init__(
        self,
        over_weight: float = 1.0,
        under_weight: float = 1.5,
        reduction: str = 'mean'
    ):
        """
        Initialize asymmetric loss.
        
        Args:
            over_weight: Weight for over-predictions (pred > target)
            under_weight: Weight for under-predictions (pred < target)
            reduction: Reduction method
        """
        super().__init__()
        
        self.over_weight = over_weight
        self.under_weight = under_weight
        self.reduction = reduction
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute asymmetric loss.
        
        Args:
            predictions: Model predictions
            targets: True values
            
        Returns:
            Asymmetric loss
        """
        errors = predictions.view(-1) - targets.view(-1)
        
        # Apply different weights based on error sign
        weights = torch.where(
            errors > 0,
            torch.tensor(self.over_weight, device=errors.device),
            torch.tensor(self.under_weight, device=errors.device)
        )
        
        weighted_errors = weights * (errors ** 2)
        
        if self.reduction == 'mean':
            return weighted_errors.mean()
        elif self.reduction == 'sum':
            return weighted_errors.sum()
        return weighted_errors


def get_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Loss function module
    """
    loss_type = config.get('type', 'huber_directional')
    
    if loss_type == 'huber_directional':
        return HuberDirectionalLoss(
            delta=config.get('huber_delta', 1.0),
            directional_weight=config.get('directional_weight', 0.1)
        )
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(
            base_weight=config.get('base_weight', 1.0),
            volatility_factor=config.get('volatility_factor', 0.5)
        )
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(
            over_weight=config.get('over_weight', 1.0),
            under_weight=config.get('under_weight', 1.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
