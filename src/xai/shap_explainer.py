"""
SHAP-based Model Explanations
=============================

Implements SHAP (SHapley Additive exPlanations) for interpreting
model predictions with temporal and feature-wise analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainer for time series forecasting models.
    
    Uses DeepExplainer for deep learning models to compute
    feature attributions efficiently.
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        feature_names: List[str],
        device: str = 'cpu'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained PyTorch model
            background_data: Background samples for SHAP [n_samples, seq_len, n_features]
            feature_names: Names of input features
            device: Device for computation
        """
        self.model = model
        self.model.eval()
        self.background_data = background_data.to(device)
        self.feature_names = feature_names
        self.device = device
        
        self.seq_len = background_data.shape[1]
        self.n_features = background_data.shape[2]
        
        # Initialize SHAP
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP DeepExplainer."""
        try:
            import shap
            
            # Wrapper function for SHAP
            def model_fn(x):
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                    output = self.model(x_tensor)
                    return output['predictions'].cpu().numpy()
            
            self.explainer = shap.DeepExplainer(
                (self.model, self._get_model_input_layer()),
                self.background_data
            )
            self._use_deep_explainer = True
            logger.info("Initialized SHAP DeepExplainer")
            
        except Exception as e:
            logger.warning(f"DeepExplainer failed, falling back to KernelExplainer: {e}")
            self._use_deep_explainer = False
            
            def model_predict(x):
                with torch.no_grad():
                    # Reshape flat input back to sequence
                    x_reshaped = x.reshape(-1, self.seq_len, self.n_features)
                    x_tensor = torch.tensor(x_reshaped, dtype=torch.float32, device=self.device)
                    output = self.model(x_tensor)
                    return output['predictions'].cpu().numpy().flatten()
            
            import shap
            # Use subset of background for KernelExplainer (slower)
            bg_flat = self.background_data[:50].cpu().numpy().reshape(50, -1)
            self.explainer = shap.KernelExplainer(model_predict, bg_flat)
    
    def _get_model_input_layer(self):
        """Get the input layer for DeepExplainer."""
        # For our model, the input goes through feature_transform first
        if hasattr(self.model, 'feature_transform'):
            return self.model.feature_transform
        return list(self.model.children())[0]
    
    def explain(
        self,
        X: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for input samples.
        
        Args:
            X: Input tensor [n_samples, seq_len, n_features]
            n_samples: Limit number of samples to explain
            
        Returns:
            Dictionary with SHAP values and related metrics
        """
        if n_samples is not None:
            X = X[:n_samples]
        
        X = X.to(self.device)
        
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        try:
            if self._use_deep_explainer:
                shap_values = self.explainer.shap_values(X)
            else:
                # Flatten for KernelExplainer
                X_flat = X.cpu().numpy().reshape(len(X), -1)
                shap_values = self.explainer.shap_values(X_flat)
                # Reshape back
                shap_values = shap_values.reshape(len(X), self.seq_len, self.n_features)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            # Fall back to gradient-based approximation
            shap_values = self._gradient_approximation(X)
        
        # Compute summary statistics
        results = {
            'shap_values': shap_values,
            'feature_importance': self._compute_feature_importance(shap_values),
            'temporal_importance': self._compute_temporal_importance(shap_values),
            'feature_names': self.feature_names
        }
        
        return results
    
    def _gradient_approximation(self, X: torch.Tensor) -> np.ndarray:
        """
        Compute gradient-based feature importance as SHAP approximation.
        
        Args:
            X: Input tensor
            
        Returns:
            Gradient-based importance values
        """
        self.model.eval()
        X = X.clone().requires_grad_(True)
        
        output = self.model(X)
        predictions = output['predictions']
        
        # Backward pass
        predictions.sum().backward()
        
        gradients = X.grad.cpu().numpy()
        
        # Scale by input values (integrated gradients approximation)
        baseline = self.background_data.mean(dim=0).cpu().numpy()
        importance = gradients * (X.detach().cpu().numpy() - baseline)
        
        return importance
    
    def _compute_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        Compute overall feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values [n_samples, seq_len, n_features]
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Mean absolute SHAP value across samples and time steps
        importance = np.abs(shap_values).mean(axis=(0, 1))
        
        # Normalize
        importance = importance / importance.sum()
        
        return dict(zip(self.feature_names, importance))
    
    def _compute_temporal_importance(self, shap_values: np.ndarray) -> np.ndarray:
        """
        Compute importance of each time step.
        
        Args:
            shap_values: SHAP values [n_samples, seq_len, n_features]
            
        Returns:
            Array of temporal importance scores [seq_len]
        """
        # Sum absolute SHAP across features, mean across samples
        temporal_importance = np.abs(shap_values).sum(axis=2).mean(axis=0)
        
        # Normalize
        temporal_importance = temporal_importance / temporal_importance.sum()
        
        return temporal_importance
    
    def explain_single(
        self,
        x: torch.Tensor,
        return_plot_data: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            x: Input tensor [1, seq_len, n_features] or [seq_len, n_features]
            return_plot_data: Whether to return data for plotting
            
        Returns:
            Dictionary with explanation details
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(x)
            prediction = output['predictions'].item()
            attention_weights = output.get('attention_weights')
        
        # Get SHAP values
        explanation = self.explain(x, n_samples=1)
        shap_values = explanation['shap_values'][0]  # [seq_len, n_features]
        
        result = {
            'prediction': prediction,
            'shap_values': shap_values,
            'feature_importance': explanation['feature_importance'],
            'temporal_importance': explanation['temporal_importance']
        }
        
        if attention_weights is not None:
            result['attention_weights'] = attention_weights.cpu().numpy()
        
        if return_plot_data:
            result['plot_data'] = self._prepare_plot_data(x[0], shap_values)
        
        return result
    
    def _prepare_plot_data(
        self,
        x: torch.Tensor,
        shap_values: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare data for visualization."""
        x_np = x.cpu().numpy()
        
        return {
            'feature_values': x_np,
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'seq_len': self.seq_len
        }


def compute_temporal_importance(
    model: nn.Module,
    data: torch.Tensor,
    feature_names: List[str],
    window_size: int = 10
) -> pd.DataFrame:
    """
    Compute feature importance over time using rolling windows.
    
    Args:
        model: Trained model
        data: Full dataset tensor [n_samples, seq_len, n_features]
        feature_names: Feature names
        window_size: Rolling window size
        
    Returns:
        DataFrame with temporal importance
    """
    n_samples = len(data)
    n_windows = n_samples - window_size + 1
    
    importances = []
    
    for i in tqdm(range(0, n_windows, window_size), desc="Computing temporal importance"):
        window_data = data[i:i + window_size]
        
        # Create explainer for this window
        explainer = SHAPExplainer(
            model,
            window_data,
            feature_names,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        explanation = explainer.explain(window_data)
        importances.append(explanation['feature_importance'])
    
    df = pd.DataFrame(importances)
    return df


def aggregate_feature_importance(
    explanations: List[Dict[str, np.ndarray]],
    feature_names: List[str],
    method: str = 'mean'
) -> Dict[str, float]:
    """
    Aggregate feature importance across multiple explanations.
    
    Args:
        explanations: List of explanation dictionaries
        feature_names: Feature names
        method: Aggregation method ('mean', 'median', 'max')
        
    Returns:
        Dictionary with aggregated importance
    """
    all_importance = []
    
    for exp in explanations:
        shap_values = exp['shap_values']
        importance = np.abs(shap_values).mean(axis=(0, 1))
        all_importance.append(importance)
    
    all_importance = np.array(all_importance)
    
    if method == 'mean':
        aggregated = all_importance.mean(axis=0)
    elif method == 'median':
        aggregated = np.median(all_importance, axis=0)
    elif method == 'max':
        aggregated = all_importance.max(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Normalize
    aggregated = aggregated / aggregated.sum()
    
    return dict(zip(feature_names, aggregated))
