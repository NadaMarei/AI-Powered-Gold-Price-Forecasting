"""
Explanation Stability Analysis
==============================

Implements stability analysis for model explanations to ensure
temporally stable and reliable interpretations.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExplanationStabilityAnalyzer:
    """
    Analyze stability of model explanations.
    
    Measures how consistent explanations are:
    1. Across small input perturbations
    2. Across different time periods
    3. Across different market regimes
    """
    
    def __init__(
        self,
        model: nn.Module,
        explainer: 'SHAPExplainer',
        feature_names: List[str]
    ):
        """
        Initialize stability analyzer.
        
        Args:
            model: Trained model
            explainer: SHAP explainer instance
            feature_names: Feature names
        """
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.device = next(model.parameters()).device
    
    def perturbation_stability(
        self,
        X: torch.Tensor,
        perturbation_std: float = 0.01,
        n_perturbations: int = 20
    ) -> Dict[str, float]:
        """
        Measure explanation stability under input perturbations.
        
        Args:
            X: Input samples [n_samples, seq_len, n_features]
            perturbation_std: Standard deviation of perturbation
            n_perturbations: Number of perturbations per sample
            
        Returns:
            Dictionary with stability metrics
        """
        X = X.to(self.device)
        n_samples = len(X)
        
        logger.info(f"Computing perturbation stability for {n_samples} samples...")
        
        # Get original explanations
        original_exp = self.explainer.explain(X)
        original_importance = original_exp['shap_values']
        
        # Storage for perturbed explanations
        all_perturbed = []
        
        for i in range(n_perturbations):
            # Add Gaussian noise
            noise = torch.randn_like(X) * perturbation_std
            X_perturbed = X + noise
            
            # Get perturbed explanations
            perturbed_exp = self.explainer.explain(X_perturbed)
            all_perturbed.append(perturbed_exp['shap_values'])
        
        all_perturbed = np.array(all_perturbed)  # [n_perturbations, n_samples, seq_len, n_features]
        
        # Compute stability metrics
        stability_metrics = {}
        
        # 1. Lipschitz stability (how much explanations change per unit input change)
        mean_explanation_change = np.mean([
            np.abs(p - original_importance).mean() 
            for p in all_perturbed
        ])
        stability_metrics['lipschitz_ratio'] = mean_explanation_change / perturbation_std
        
        # 2. Rank correlation stability
        original_ranks = self._rank_features(original_importance)
        perturbed_ranks = [self._rank_features(p) for p in all_perturbed]
        
        rank_correlations = [
            self._spearman_correlation(original_ranks, pr)
            for pr in perturbed_ranks
        ]
        stability_metrics['rank_correlation_mean'] = np.mean(rank_correlations)
        stability_metrics['rank_correlation_std'] = np.std(rank_correlations)
        
        # 3. Top-k stability (do top features remain top?)
        k = min(5, len(self.feature_names))
        top_k_stability = self._top_k_stability(original_importance, all_perturbed, k)
        stability_metrics['top_k_stability'] = top_k_stability
        
        # 4. Sign stability (do positive/negative attributions stay consistent?)
        sign_stability = self._sign_stability(original_importance, all_perturbed)
        stability_metrics['sign_stability'] = sign_stability
        
        return stability_metrics
    
    def temporal_stability(
        self,
        X: torch.Tensor,
        window_size: int = 20
    ) -> Dict[str, float]:
        """
        Measure stability of explanations over time.
        
        Args:
            X: Time series data [n_samples, seq_len, n_features]
            window_size: Size of temporal windows
            
        Returns:
            Dictionary with temporal stability metrics
        """
        n_samples = len(X)
        n_windows = n_samples // window_size
        
        if n_windows < 2:
            logger.warning("Not enough data for temporal stability analysis")
            return {}
        
        window_importances = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = X[start_idx:end_idx].to(self.device)
            
            explanation = self.explainer.explain(window_data)
            # Aggregate importance for this window
            importance = np.abs(explanation['shap_values']).mean(axis=(0, 1))
            window_importances.append(importance)
        
        window_importances = np.array(window_importances)
        
        # Metrics
        stability_metrics = {}
        
        # 1. Coefficient of variation for each feature
        cv = np.std(window_importances, axis=0) / (np.mean(window_importances, axis=0) + 1e-8)
        stability_metrics['mean_cv'] = np.mean(cv)
        stability_metrics['max_cv'] = np.max(cv)
        
        # 2. Rank correlation between consecutive windows
        rank_corrs = []
        for i in range(len(window_importances) - 1):
            corr = stats.spearmanr(window_importances[i], window_importances[i + 1])[0]
            rank_corrs.append(corr)
        
        stability_metrics['temporal_rank_correlation'] = np.mean(rank_corrs)
        
        # 3. Consistency of top features
        top_k = 5
        top_features_per_window = [
            set(np.argsort(imp)[-top_k:])
            for imp in window_importances
        ]
        
        # Jaccard similarity between consecutive windows
        jaccard_scores = []
        for i in range(len(top_features_per_window) - 1):
            intersection = len(top_features_per_window[i] & top_features_per_window[i + 1])
            union = len(top_features_per_window[i] | top_features_per_window[i + 1])
            jaccard_scores.append(intersection / union)
        
        stability_metrics['top_feature_consistency'] = np.mean(jaccard_scores)
        
        return stability_metrics
    
    def regime_stability(
        self,
        X: torch.Tensor,
        volatility: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Compare explanations across different market regimes.
        
        Args:
            X: Input data
            volatility: Volatility indicator for regime detection
            threshold: Threshold for regime classification
            
        Returns:
            Dictionary with regime stability metrics
        """
        X = X.to(self.device)
        
        # Split into regimes
        low_vol_mask = volatility < threshold
        high_vol_mask = ~low_vol_mask
        
        if low_vol_mask.sum() < 10 or high_vol_mask.sum() < 10:
            logger.warning("Insufficient samples in one regime")
            return {}
        
        # Get explanations for each regime
        low_vol_exp = self.explainer.explain(X[low_vol_mask])
        high_vol_exp = self.explainer.explain(X[high_vol_mask])
        
        # Feature importance comparison
        low_vol_importance = low_vol_exp['feature_importance']
        high_vol_importance = high_vol_exp['feature_importance']
        
        # Rank correlation between regimes
        low_vol_ranks = np.argsort(list(low_vol_importance.values()))
        high_vol_ranks = np.argsort(list(high_vol_importance.values()))
        rank_correlation = stats.spearmanr(low_vol_ranks, high_vol_ranks)[0]
        
        # Feature importance difference
        importance_diff = {
            feat: high_vol_importance[feat] - low_vol_importance[feat]
            for feat in self.feature_names
        }
        
        return {
            'rank_correlation': rank_correlation,
            'low_vol_importance': low_vol_importance,
            'high_vol_importance': high_vol_importance,
            'importance_difference': importance_diff,
            'regime_dependent_features': [
                feat for feat, diff in importance_diff.items()
                if abs(diff) > 0.1
            ]
        }
    
    def _rank_features(self, importance: np.ndarray) -> np.ndarray:
        """Convert importance values to ranks."""
        # Aggregate across samples and time
        agg_importance = np.abs(importance).mean(axis=(0, 1))
        return stats.rankdata(agg_importance)
    
    def _spearman_correlation(self, ranks1: np.ndarray, ranks2: np.ndarray) -> float:
        """Compute Spearman rank correlation."""
        return stats.spearmanr(ranks1, ranks2)[0]
    
    def _top_k_stability(
        self,
        original: np.ndarray,
        perturbed: np.ndarray,
        k: int
    ) -> float:
        """Compute stability of top-k features."""
        original_agg = np.abs(original).mean(axis=(0, 1))
        original_top_k = set(np.argsort(original_agg)[-k:])
        
        stability_scores = []
        for p in perturbed:
            p_agg = np.abs(p).mean(axis=(0, 1))
            p_top_k = set(np.argsort(p_agg)[-k:])
            
            # Jaccard similarity
            intersection = len(original_top_k & p_top_k)
            union = len(original_top_k | p_top_k)
            stability_scores.append(intersection / union)
        
        return np.mean(stability_scores)
    
    def _sign_stability(
        self,
        original: np.ndarray,
        perturbed: np.ndarray
    ) -> float:
        """Compute stability of attribution signs."""
        original_signs = np.sign(original.mean(axis=(0, 1)))
        
        sign_match_rates = []
        for p in perturbed:
            p_signs = np.sign(p.mean(axis=(0, 1)))
            match_rate = np.mean(original_signs == p_signs)
            sign_match_rates.append(match_rate)
        
        return np.mean(sign_match_rates)


def compute_stability_metrics(
    model: nn.Module,
    explainer: 'SHAPExplainer',
    X: torch.Tensor,
    feature_names: List[str],
    volatility: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive stability metrics.
    
    Args:
        model: Trained model
        explainer: SHAP explainer
        X: Input data
        feature_names: Feature names
        volatility: Optional volatility data for regime analysis
        
    Returns:
        Dictionary with all stability metrics
    """
    analyzer = ExplanationStabilityAnalyzer(model, explainer, feature_names)
    
    # Perturbation stability
    perturbation_metrics = analyzer.perturbation_stability(X)
    
    # Temporal stability
    temporal_metrics = analyzer.temporal_stability(X)
    
    # Regime stability
    if volatility is not None:
        regime_metrics = analyzer.regime_stability(X, volatility)
    else:
        regime_metrics = {}
    
    # Combine all metrics
    all_metrics = {
        **{f'perturbation_{k}': v for k, v in perturbation_metrics.items()},
        **{f'temporal_{k}': v for k, v in temporal_metrics.items()},
        **{f'regime_{k}': v for k, v in regime_metrics.items() if isinstance(v, (int, float))}
    }
    
    # Overall stability score (weighted average)
    if all_metrics:
        stability_score = (
            0.4 * all_metrics.get('perturbation_rank_correlation_mean', 0.5) +
            0.3 * all_metrics.get('temporal_rank_correlation', 0.5) +
            0.3 * all_metrics.get('perturbation_top_k_stability', 0.5)
        )
        all_metrics['overall_stability_score'] = stability_score
    
    return all_metrics
