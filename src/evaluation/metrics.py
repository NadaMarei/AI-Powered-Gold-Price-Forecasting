"""
Evaluation Metrics for Gold Price Forecasting
==============================================

Comprehensive metrics for evaluating forecast accuracy including:
- Standard regression metrics (RMSE, MAE, MAPE)
- Financial metrics (directional accuracy, profit metrics)
- Bootstrap confidence intervals
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics."""
    
    # Primary metrics
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    
    # Confidence intervals
    rmse_ci: Tuple[float, float] = (0.0, 0.0)
    mae_ci: Tuple[float, float] = (0.0, 0.0)
    mape_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Directional metrics
    directional_accuracy: float = 0.0
    directional_accuracy_ci: Tuple[float, float] = (0.0, 0.0)
    
    # Additional metrics
    r_squared: float = 0.0
    mse: float = 0.0
    median_ae: float = 0.0
    max_error: float = 0.0
    
    # Error distribution
    error_mean: float = 0.0
    error_std: float = 0.0
    error_skewness: float = 0.0
    error_kurtosis: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'rmse_ci': self.rmse_ci,
            'mae_ci': self.mae_ci,
            'mape_ci': self.mape_ci,
            'directional_accuracy': self.directional_accuracy,
            'directional_accuracy_ci': self.directional_accuracy_ci,
            'r_squared': self.r_squared,
            'mse': self.mse,
            'median_ae': self.median_ae,
            'max_error': self.max_error,
            'error_mean': self.error_mean,
            'error_std': self.error_std,
            'error_skewness': self.error_skewness,
            'error_kurtosis': self.error_kurtosis
        }


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def compute_mape(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error."""
    return np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100


def compute_directional_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    previous_values: np.ndarray
) -> float:
    """
    Compute directional accuracy (hit rate).
    
    Args:
        predictions: Model predictions
        targets: True values
        previous_values: Previous time step values
        
    Returns:
        Directional accuracy between 0 and 1
    """
    pred_direction = np.sign(predictions - previous_values)
    actual_direction = np.sign(targets - previous_values)
    return np.mean(pred_direction == actual_direction)


def compute_r_squared(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_confidence_intervals(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        predictions: Model predictions
        targets: True values
        metric_fn: Function to compute metric (predictions, targets) -> float
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_state: Random seed
        
    Returns:
        Tuple of (point_estimate, (lower_ci, upper_ci))
    """
    np.random.seed(random_state)
    n_samples = len(predictions)
    
    # Point estimate
    point_estimate = metric_fn(predictions, targets)
    
    # Bootstrap resampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        boot_pred = predictions[indices]
        boot_target = targets[indices]
        boot_metric = metric_fn(boot_pred, boot_target)
        bootstrap_metrics.append(boot_metric)
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
    
    return point_estimate, (lower, upper)


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    previous_values: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> ForecastMetrics:
    """
    Compute all forecast metrics with confidence intervals.
    
    Args:
        predictions: Model predictions
        targets: True values
        previous_values: Previous time step values for directional accuracy
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        ForecastMetrics object with all computed metrics
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    
    # Compute errors
    errors = predictions - targets
    
    # Primary metrics with confidence intervals
    rmse, rmse_ci = bootstrap_confidence_intervals(
        predictions, targets, compute_rmse, n_bootstrap, confidence_level
    )
    
    mae, mae_ci = bootstrap_confidence_intervals(
        predictions, targets, compute_mae, n_bootstrap, confidence_level
    )
    
    mape, mape_ci = bootstrap_confidence_intervals(
        predictions, targets, compute_mape, n_bootstrap, confidence_level
    )
    
    # Directional accuracy
    if previous_values is not None:
        previous_values = np.asarray(previous_values).flatten()
        
        def dir_acc_fn(pred, tgt):
            return compute_directional_accuracy(pred, tgt, previous_values[:len(pred)])
        
        directional_accuracy, dir_acc_ci = bootstrap_confidence_intervals(
            predictions, targets, dir_acc_fn, n_bootstrap, confidence_level
        )
    else:
        directional_accuracy = 0.0
        dir_acc_ci = (0.0, 0.0)
    
    # Additional metrics
    r_squared = compute_r_squared(predictions, targets)
    mse = np.mean(errors ** 2)
    median_ae = np.median(np.abs(errors))
    max_error = np.max(np.abs(errors))
    
    # Error distribution statistics
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_skewness = stats.skew(errors) if len(errors) > 2 else 0.0
    error_kurtosis = stats.kurtosis(errors) if len(errors) > 3 else 0.0
    
    return ForecastMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        rmse_ci=rmse_ci,
        mae_ci=mae_ci,
        mape_ci=mape_ci,
        directional_accuracy=directional_accuracy,
        directional_accuracy_ci=dir_acc_ci,
        r_squared=r_squared,
        mse=mse,
        median_ae=median_ae,
        max_error=max_error,
        error_mean=error_mean,
        error_std=error_std,
        error_skewness=error_skewness,
        error_kurtosis=error_kurtosis
    )


def compute_metrics_by_regime(
    predictions: np.ndarray,
    targets: np.ndarray,
    volatility: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, ForecastMetrics]:
    """
    Compute metrics separately for different volatility regimes.
    
    Args:
        predictions: Model predictions
        targets: True values
        volatility: Volatility indicators
        threshold: Threshold to split low/high volatility
        
    Returns:
        Dictionary with metrics for each regime
    """
    low_vol_mask = volatility < threshold
    high_vol_mask = ~low_vol_mask
    
    results = {}
    
    if low_vol_mask.sum() > 10:
        results['low_volatility'] = compute_all_metrics(
            predictions[low_vol_mask],
            targets[low_vol_mask],
            n_bootstrap=500
        )
    
    if high_vol_mask.sum() > 10:
        results['high_volatility'] = compute_all_metrics(
            predictions[high_vol_mask],
            targets[high_vol_mask],
            n_bootstrap=500
        )
    
    return results


def compute_temporal_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    window_size: int = 20
) -> Dict[str, np.ndarray]:
    """
    Compute rolling window metrics over time.
    
    Args:
        predictions: Model predictions
        targets: True values
        dates: Corresponding dates
        window_size: Rolling window size
        
    Returns:
        Dictionary with rolling metrics arrays
    """
    n_samples = len(predictions)
    n_windows = n_samples - window_size + 1
    
    rolling_rmse = np.zeros(n_windows)
    rolling_mae = np.zeros(n_windows)
    rolling_dir_acc = np.zeros(n_windows)
    
    for i in range(n_windows):
        window_pred = predictions[i:i + window_size]
        window_target = targets[i:i + window_size]
        
        rolling_rmse[i] = compute_rmse(window_pred, window_target)
        rolling_mae[i] = compute_mae(window_pred, window_target)
        
        if i > 0:
            prev_values = targets[i - 1:i + window_size - 1]
            rolling_dir_acc[i] = compute_directional_accuracy(
                window_pred, window_target, prev_values
            )
    
    return {
        'dates': dates[window_size - 1:],
        'rolling_rmse': rolling_rmse,
        'rolling_mae': rolling_mae,
        'rolling_dir_acc': rolling_dir_acc
    }
