"""
Counterfactual Analysis
=======================

Generates counterfactual explanations for gold price forecasting,
enabling "what-if" analysis for key economic scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Generate counterfactual explanations for model predictions.
    
    Creates "what-if" scenarios by modifying input features and
    observing how predictions change.
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize counterfactual generator.
        
        Args:
            model: Trained model
            feature_names: Names of input features
            feature_ranges: Valid ranges for each feature
            device: Device for computation
        """
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}
        self.device = device
        
        self.feature_idx = {name: i for i, name in enumerate(feature_names)}
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        target_prediction: float,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        feature_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate a counterfactual that achieves a target prediction.
        
        Args:
            x: Original input [seq_len, n_features] or [1, seq_len, n_features]
            target_prediction: Desired prediction value
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            feature_weights: Weights penalizing changes to each feature
            
        Returns:
            Dictionary with counterfactual and analysis
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        x_cf = x.clone().requires_grad_(True)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(x)
            original_pred = original_output['predictions'].item()
        
        # Optimization
        optimizer = torch.optim.Adam([x_cf], lr=learning_rate)
        
        # Feature change weights
        weights = torch.ones(len(self.feature_names), device=self.device)
        if feature_weights:
            for feat, weight in feature_weights.items():
                if feat in self.feature_idx:
                    weights[self.feature_idx[feat]] = weight
        
        best_cf = x_cf.clone()
        best_distance = float('inf')
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(x_cf)
            pred = output['predictions']
            
            # Loss: prediction error + distance penalty
            pred_loss = (pred - target_prediction) ** 2
            
            # Weighted L2 distance from original
            diff = x_cf - x
            distance_loss = (diff ** 2 * weights.view(1, 1, -1)).sum()
            
            loss = pred_loss + 0.1 * distance_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Apply feature constraints
            with torch.no_grad():
                x_cf.data = self._apply_constraints(x_cf.data, x.data)
            
            # Track best
            current_distance = torch.abs(pred - target_prediction).item()
            if current_distance < best_distance:
                best_distance = current_distance
                best_cf = x_cf.clone()
            
            # Early stopping
            if current_distance < 0.001:
                break
        
        # Analyze counterfactual
        with torch.no_grad():
            cf_output = self.model(best_cf)
            cf_pred = cf_output['predictions'].item()
        
        changes = self._analyze_changes(x[0], best_cf[0])
        
        return {
            'original': x[0].cpu().numpy(),
            'counterfactual': best_cf[0].detach().cpu().numpy(),
            'original_prediction': original_pred,
            'counterfactual_prediction': cf_pred,
            'target_prediction': target_prediction,
            'feature_changes': changes,
            'convergence': best_distance < 0.01
        }
    
    def _apply_constraints(
        self,
        x: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """Apply feature range constraints."""
        for feat, (min_val, max_val) in self.feature_ranges.items():
            if feat in self.feature_idx:
                idx = self.feature_idx[feat]
                x[:, :, idx] = torch.clamp(x[:, :, idx], min_val, max_val)
        
        return x
    
    def _analyze_changes(
        self,
        original: torch.Tensor,
        counterfactual: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Analyze changes between original and counterfactual."""
        original = original.cpu().numpy()
        counterfactual = counterfactual.detach().cpu().numpy()
        
        changes = {}
        for i, feat in enumerate(self.feature_names):
            orig_mean = original[:, i].mean()
            cf_mean = counterfactual[:, i].mean()
            
            changes[feat] = {
                'original_mean': orig_mean,
                'counterfactual_mean': cf_mean,
                'absolute_change': cf_mean - orig_mean,
                'percent_change': (cf_mean - orig_mean) / (orig_mean + 1e-8) * 100
            }
        
        return changes
    
    def scenario_analysis(
        self,
        x: torch.Tensor,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple predefined scenarios.
        
        Args:
            x: Base input
            scenarios: Dictionary of scenario names to feature modifications
                      e.g., {'high_volatility': {'Volatility_20': 2.0}}
            
        Returns:
            Dictionary with results for each scenario
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(x)
            baseline_pred = baseline_output['predictions'].item()
        
        results = {
            'baseline': {
                'prediction': baseline_pred,
                'input': x[0].cpu().numpy()
            }
        }
        
        for scenario_name, modifications in scenarios.items():
            x_scenario = x.clone()
            
            # Apply modifications
            for feat, value in modifications.items():
                if feat in self.feature_idx:
                    idx = self.feature_idx[feat]
                    # Modify last time step (most recent)
                    x_scenario[0, -1, idx] = value
            
            # Get scenario prediction
            with torch.no_grad():
                scenario_output = self.model(x_scenario)
                scenario_pred = scenario_output['predictions'].item()
            
            results[scenario_name] = {
                'prediction': scenario_pred,
                'prediction_change': scenario_pred - baseline_pred,
                'percent_change': (scenario_pred - baseline_pred) / baseline_pred * 100,
                'modifications': modifications
            }
        
        return results


def generate_economic_scenarios(
    model: nn.Module,
    x: torch.Tensor,
    feature_names: List[str],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Generate standard economic scenario analysis.
    
    Analyzes model predictions under various economic conditions
    relevant to gold price forecasting.
    
    Args:
        model: Trained model
        x: Base input
        feature_names: Feature names
        device: Device for computation
        
    Returns:
        Dictionary with scenario analysis results
    """
    generator = CounterfactualGenerator(model, feature_names, device=device)
    
    # Define economic scenarios
    scenarios = {}
    
    # Volatility scenarios
    if 'Volatility_20' in feature_names:
        scenarios['high_volatility'] = {'Volatility_20': 0.3}  # Annualized 30%
        scenarios['low_volatility'] = {'Volatility_20': 0.1}   # Annualized 10%
    
    # Return scenarios
    if 'Returns' in feature_names:
        scenarios['positive_momentum'] = {'Returns': 0.02}  # 2% daily return
        scenarios['negative_momentum'] = {'Returns': -0.02}
    
    # Dollar scenarios (if available)
    if 'DXY' in feature_names:
        scenarios['dollar_strength'] = {'DXY': 110}  # Strong dollar
        scenarios['dollar_weakness'] = {'DXY': 90}   # Weak dollar
    
    # VIX scenarios (if available)
    if 'VIX' in feature_names:
        scenarios['market_fear'] = {'VIX': 35}       # High fear
        scenarios['market_complacency'] = {'VIX': 12} # Low fear
    
    # Run scenario analysis
    results = generator.scenario_analysis(x, scenarios)
    
    # Add interpretation
    interpretations = {}
    baseline_pred = results['baseline']['prediction']
    
    for scenario, data in results.items():
        if scenario == 'baseline':
            continue
        
        pred_change = data['prediction_change']
        
        if abs(pred_change) < baseline_pred * 0.01:
            interpretation = "Minimal impact on prediction"
        elif pred_change > 0:
            interpretation = f"Increases gold price prediction by {data['percent_change']:.2f}%"
        else:
            interpretation = f"Decreases gold price prediction by {abs(data['percent_change']):.2f}%"
        
        interpretations[scenario] = interpretation
    
    results['interpretations'] = interpretations
    
    return results
