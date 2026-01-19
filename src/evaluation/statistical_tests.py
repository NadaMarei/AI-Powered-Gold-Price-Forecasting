"""
Statistical Tests for Model Comparison
======================================

Implements rigorous statistical tests for comparing forecast models:
- Diebold-Mariano test for forecast accuracy
- Model Confidence Set
- Granger causality tests
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss_type: str = 'squared',
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Tests the null hypothesis that the two forecasts have equal accuracy
    against the alternative that forecast 1 is more accurate.
    
    Args:
        errors_1: Forecast errors from model 1
        errors_2: Forecast errors from model 2
        h: Forecast horizon
        loss_type: 'squared' for MSE-based, 'absolute' for MAE-based
        alternative: 'two-sided', 'less' (model 1 better), 'greater' (model 2 better)
        
    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    n = len(errors_1)
    
    # Compute loss differential
    if loss_type == 'squared':
        d = errors_1 ** 2 - errors_2 ** 2
    elif loss_type == 'absolute':
        d = np.abs(errors_1) - np.abs(errors_2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Mean loss differential
    d_bar = np.mean(d)
    
    # Autocovariance estimation for variance
    # Using Newey-West HAC estimator for h-step ahead forecasts
    gamma_0 = np.var(d, ddof=1)
    
    if h > 1:
        # Compute autocovariances
        gamma = np.zeros(h)
        for k in range(h):
            gamma[k] = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)) if k > 0 else gamma_0
        
        # HAC variance estimator
        var_d = gamma_0 + 2 * sum(gamma[1:h])
    else:
        var_d = gamma_0
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d / n)
    
    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    elif alternative == 'less':  # Model 1 is better
        p_value = stats.norm.cdf(dm_stat)
    else:  # Model 2 is better
        p_value = 1 - stats.norm.cdf(dm_stat)
    
    # Interpretation
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.10:
        significance = "*"
    else:
        significance = ""
    
    conclusion = ""
    if p_value < 0.05:
        if d_bar < 0:
            conclusion = "Model 1 significantly outperforms Model 2"
        else:
            conclusion = "Model 2 significantly outperforms Model 1"
    else:
        conclusion = "No significant difference between models"
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mean_loss_diff': d_bar,
        'significance': significance,
        'conclusion': conclusion,
        'loss_type': loss_type,
        'alternative': alternative
    }


def modified_diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1
) -> Dict[str, float]:
    """
    Harvey, Leybourne and Newbold (1997) modified DM test.
    
    Uses a t-distribution for small samples and includes a
    correction factor for improved finite sample performance.
    
    Args:
        errors_1: Forecast errors from model 1
        errors_2: Forecast errors from model 2
        h: Forecast horizon
        
    Returns:
        Dictionary with test results
    """
    n = len(errors_1)
    
    # Loss differential
    d = errors_1 ** 2 - errors_2 ** 2
    d_bar = np.mean(d)
    
    # Variance estimation
    var_d = np.var(d, ddof=1)
    
    # Modified DM statistic with small sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = d_bar / (np.sqrt(var_d / n) * correction)
    
    # Use t-distribution with n-1 degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_star), df=n - 1))
    
    return {
        'dm_statistic': dm_star,
        'p_value': p_value,
        'mean_loss_diff': d_bar,
        'degrees_of_freedom': n - 1
    }


def model_confidence_set(
    losses: Dict[str, np.ndarray],
    alpha: float = 0.1,
    n_bootstrap: int = 1000
) -> Dict[str, any]:
    """
    Hansen's Model Confidence Set (MCS) procedure.
    
    Identifies a set of models that contains the best model
    with a given confidence level.
    
    Args:
        losses: Dictionary mapping model names to loss arrays
        alpha: Significance level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with MCS results
    """
    model_names = list(losses.keys())
    n_models = len(model_names)
    n_obs = len(list(losses.values())[0])
    
    # Create loss matrix
    loss_matrix = np.column_stack([losses[name] for name in model_names])
    
    # Initialize surviving models
    surviving = set(range(n_models))
    eliminated = []
    
    np.random.seed(42)
    
    while len(surviving) > 1:
        # Compute relative losses for surviving models
        surv_list = list(surviving)
        surv_losses = loss_matrix[:, surv_list]
        mean_losses = surv_losses.mean(axis=0)
        
        # Test statistic: max relative loss
        rel_losses = mean_losses - mean_losses.min()
        t_stat = rel_losses.max()
        
        # Bootstrap p-value
        boot_stats = []
        for _ in range(n_bootstrap):
            boot_idx = np.random.randint(0, n_obs, n_obs)
            boot_losses = surv_losses[boot_idx]
            boot_means = boot_losses.mean(axis=0)
            boot_rel = boot_means - boot_means.min()
            boot_stats.append(boot_rel.max())
        
        p_value = np.mean(np.array(boot_stats) >= t_stat)
        
        if p_value >= alpha:
            # Cannot reject equality, stop elimination
            break
        
        # Eliminate worst model
        worst_idx = np.argmax(rel_losses)
        eliminated.append(model_names[surv_list[worst_idx]])
        surviving.remove(surv_list[worst_idx])
    
    # Final MCS
    mcs_models = [model_names[i] for i in surviving]
    
    return {
        'mcs_models': mcs_models,
        'eliminated_models': eliminated,
        'n_surviving': len(mcs_models),
        'alpha': alpha
    }


def granger_causality_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int = 5
) -> Dict[str, any]:
    """
    Granger causality test.
    
    Tests if x Granger-causes y (i.e., past values of x help
    predict y beyond past values of y alone).
    
    Args:
        y: Target variable
        x: Potential causal variable
        max_lag: Maximum lag to consider
        
    Returns:
        Dictionary with test results for each lag
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Prepare data
    data = np.column_stack([y, x])
    
    results = {}
    
    try:
        gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        for lag in range(1, max_lag + 1):
            f_stat = gc_results[lag][0]['ssr_ftest'][0]
            p_value = gc_results[lag][0]['ssr_ftest'][1]
            
            results[f'lag_{lag}'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    except Exception as e:
        logger.warning(f"Granger causality test failed: {e}")
        return {'error': str(e)}
    
    return results


def forecast_encompassing_test(
    targets: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray
) -> Dict[str, float]:
    """
    Forecast encompassing test (Harvey-Leybourne-Newbold).
    
    Tests if forecast 1 encompasses forecast 2 (i.e., forecast 2
    adds no information beyond what's in forecast 1).
    
    Args:
        targets: Actual values
        pred_1: Predictions from model 1
        pred_2: Predictions from model 2
        
    Returns:
        Dictionary with test results
    """
    from scipy import stats
    
    e1 = targets - pred_1
    e2 = targets - pred_2
    
    # Regression: e1 = alpha + beta * (e1 - e2) + u
    d = e1 - e2
    
    # OLS estimation
    n = len(e1)
    x = np.column_stack([np.ones(n), d])
    beta = np.linalg.lstsq(x, e1, rcond=None)[0]
    
    residuals = e1 - x @ beta
    sigma2 = np.sum(residuals ** 2) / (n - 2)
    
    # Standard error of slope
    var_beta = sigma2 * np.linalg.inv(x.T @ x)
    se_slope = np.sqrt(var_beta[1, 1])
    
    # t-statistic for H0: beta = 1
    t_stat = (beta[1] - 1) / se_slope
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
    
    return {
        'coefficient': beta[1],
        't_statistic': t_stat,
        'p_value': p_value,
        'forecast_1_encompasses_2': p_value > 0.05 and beta[1] > 0.5
    }


def white_reality_check(
    benchmark_losses: np.ndarray,
    model_losses: Dict[str, np.ndarray],
    n_bootstrap: int = 1000
) -> Dict[str, float]:
    """
    White's Reality Check for data snooping.
    
    Tests if any model significantly outperforms the benchmark
    after accounting for multiple testing.
    
    Args:
        benchmark_losses: Losses from benchmark model
        model_losses: Dictionary of losses from candidate models
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with test results
    """
    model_names = list(model_losses.keys())
    n_obs = len(benchmark_losses)
    
    # Compute relative performance
    f_bar = {}
    for name, losses in model_losses.items():
        f_bar[name] = np.mean(benchmark_losses - losses)
    
    # Maximum relative performance
    max_f_bar = max(f_bar.values())
    
    # Bootstrap distribution of max relative performance
    np.random.seed(42)
    boot_max = []
    
    for _ in range(n_bootstrap):
        boot_idx = np.random.randint(0, n_obs, n_obs)
        boot_f = {}
        for name, losses in model_losses.items():
            boot_f[name] = np.mean(benchmark_losses[boot_idx] - losses[boot_idx])
        boot_max.append(max(boot_f.values()))
    
    # P-value
    p_value = np.mean(np.array(boot_max) >= max_f_bar)
    
    # Best model
    best_model = max(f_bar, key=f_bar.get)
    
    return {
        'p_value': p_value,
        'significant': p_value < 0.05,
        'best_model': best_model,
        'best_improvement': f_bar[best_model],
        'all_improvements': f_bar
    }
