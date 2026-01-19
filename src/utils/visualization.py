"""
Visualization Utilities
=======================

Publication-quality visualization for:
- Training curves
- XAI explanations
- Model comparison
- Performance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})


class TrainingVisualizer:
    """Visualize training progress and convergence."""
    
    def __init__(self, save_dir: str = "figures"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        history: Dict[str, List],
        model_name: str = "model",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Training history dictionary
            model_name: Name for the plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.set_yscale('log')
        
        # Learning rate
        ax = axes[0, 1]
        ax.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        
        # Validation metrics
        ax = axes[1, 0]
        val_metrics = history.get('val_metrics', [])
        if val_metrics:
            rmse = [m.get('rmse', 0) for m in val_metrics]
            mae = [m.get('mae', 0) for m in val_metrics]
            ax.plot(epochs, rmse, 'b-', label='RMSE', linewidth=2)
            ax.plot(epochs, mae, 'r-', label='MAE', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Error')
            ax.set_title('Validation Metrics')
            ax.legend()
        
        # Directional accuracy
        ax = axes[1, 1]
        if val_metrics:
            dir_acc = [m.get('directional_accuracy', 0) for m in val_metrics]
            ax.plot(epochs, dir_acc, 'purple', linewidth=2)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Directional Accuracy')
            ax.set_title('Directional Accuracy')
            ax.set_ylim([0.3, 0.8])
            ax.legend()
        
        plt.suptitle(f'Training Progress: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / f'{model_name}_training_curves.png')
            fig.savefig(self.save_dir / f'{model_name}_training_curves.pdf')
        
        return fig
    
    def plot_convergence_comparison(
        self,
        histories: Dict[str, Dict[str, List]],
        save: bool = True
    ) -> plt.Figure:
        """
        Compare convergence across multiple models.
        
        Args:
            histories: Dictionary of model names to history dicts
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
        
        for (name, history), color in zip(histories.items(), colors):
            epochs = range(1, len(history['train_loss']) + 1)
            
            axes[0].plot(epochs, history['val_loss'], label=name, color=color, linewidth=2)
            
            val_metrics = history.get('val_metrics', [])
            if val_metrics:
                rmse = [m.get('rmse', 0) for m in val_metrics]
                axes[1].plot(epochs, rmse, label=name, color=color, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Convergence Comparison')
        axes[0].legend()
        axes[0].set_yscale('log')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation RMSE')
        axes[1].set_title('RMSE Over Training')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'convergence_comparison.png')
            fig.savefig(self.save_dir / 'convergence_comparison.pdf')
        
        return fig


class XAIVisualizer:
    """Visualize model explanations."""
    
    def __init__(self, save_dir: str = "figures"):
        """
        Initialize XAI visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        title: str = "Feature Importance",
        top_k: int = 20,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance: Dictionary of feature names to importance values
            title: Plot title
            top_k: Number of top features to show
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Sort and select top features
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, values = zip(*sorted_importance)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        bars = ax.barh(range(len(features)), values, color=colors)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filename = title.lower().replace(' ', '_')
            fig.savefig(self.save_dir / f'{filename}.png')
            fig.savefig(self.save_dir / f'{filename}.pdf')
        
        return fig
    
    def plot_temporal_importance_heatmap(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        title: str = "Temporal Feature Importance",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot heatmap of SHAP values over time.
        
        Args:
            shap_values: SHAP values [seq_len, n_features]
            feature_names: Feature names
            title: Plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create diverging colormap
        cmap = plt.cm.RdBu_r
        
        # Plot heatmap
        im = ax.imshow(shap_values.T, aspect='auto', cmap=cmap, 
                       vmin=-np.abs(shap_values).max(), vmax=np.abs(shap_values).max())
        
        # Labels
        ax.set_xlabel('Time Step (days from prediction)')
        ax.set_ylabel('Feature')
        ax.set_title(title, fontweight='bold')
        
        # Set y-axis labels
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SHAP Value')
        
        plt.tight_layout()
        
        if save:
            filename = title.lower().replace(' ', '_')
            fig.savefig(self.save_dir / f'{filename}.png')
            fig.savefig(self.save_dir / f'{filename}.pdf')
        
        return fig
    
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        title: str = "Temporal Attention Weights",
        save: bool = True
    ) -> plt.Figure:
        """
        Plot attention weights over time steps.
        
        Args:
            attention_weights: Attention weights [n_heads, seq_len]
            title: Plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Average attention across heads
        avg_attention = attention_weights.mean(axis=0)
        seq_len = len(avg_attention)
        
        # Line plot
        ax = axes[0]
        ax.plot(range(seq_len), avg_attention, 'b-', linewidth=2)
        ax.fill_between(range(seq_len), avg_attention, alpha=0.3)
        ax.set_xlabel('Time Step (days from prediction)')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Average Attention Distribution')
        
        # Heatmap for all heads
        ax = axes[1]
        im = ax.imshow(attention_weights, aspect='auto', cmap='Blues')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Head')
        ax.set_title('Per-Head Attention Weights')
        plt.colorbar(im, ax=ax, label='Weight')
        
        plt.suptitle(title, fontweight='bold', fontsize=12)
        plt.tight_layout()
        
        if save:
            filename = title.lower().replace(' ', '_')
            fig.savefig(self.save_dir / f'{filename}.png')
            fig.savefig(self.save_dir / f'{filename}.pdf')
        
        return fig
    
    def plot_stability_analysis(
        self,
        stability_metrics: Dict[str, float],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot explanation stability metrics.
        
        Args:
            stability_metrics: Dictionary of stability metrics
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Perturbation stability
        perturbation_metrics = {
            k.replace('perturbation_', ''): v 
            for k, v in stability_metrics.items() 
            if k.startswith('perturbation_')
        }
        
        if perturbation_metrics:
            ax = axes[0]
            keys = list(perturbation_metrics.keys())
            values = list(perturbation_metrics.values())
            ax.barh(keys, values, color='steelblue')
            ax.set_xlabel('Value')
            ax.set_title('Perturbation Stability')
        
        # Temporal stability
        temporal_metrics = {
            k.replace('temporal_', ''): v 
            for k, v in stability_metrics.items() 
            if k.startswith('temporal_')
        }
        
        if temporal_metrics:
            ax = axes[1]
            keys = list(temporal_metrics.keys())
            values = list(temporal_metrics.values())
            ax.barh(keys, values, color='darkgreen')
            ax.set_xlabel('Value')
            ax.set_title('Temporal Stability')
        
        # Overall stability score
        ax = axes[2]
        overall_score = stability_metrics.get('overall_stability_score', 0)
        ax.pie([overall_score, 1 - overall_score], 
               labels=['Stable', 'Unstable'],
               colors=['green', 'red'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f'Overall Stability: {overall_score:.2f}')
        
        plt.suptitle('Explanation Stability Analysis', fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'stability_analysis.png')
            fig.savefig(self.save_dir / 'stability_analysis.pdf')
        
        return fig


class ResultsVisualizer:
    """Visualize model comparison results."""
    
    def __init__(self, save_dir: str = "figures"):
        """
        Initialize results visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['rmse', 'mae', 'directional_accuracy'],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot comparison of models across metrics.
        
        Args:
            results: Dictionary of model names to metric dictionaries
            metrics: Metrics to compare
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        models = list(results.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for ax, metric in zip(axes, metrics):
            values = [results[m].get(metric, 0) for m in models]
            
            # Handle confidence intervals if available
            ci_lower = [results[m].get(f'{metric}_ci', (0, 0))[0] for m in models]
            ci_upper = [results[m].get(f'{metric}_ci', (0, 0))[1] for m in models]
            
            bars = ax.bar(models, values, color=colors)
            
            # Add error bars if CI available
            if any(ci_lower) and any(ci_upper):
                yerr_lower = [v - l for v, l in zip(values, ci_lower)]
                yerr_upper = [u - v for v, u in zip(values, ci_upper)]
                ax.errorbar(models, values, yerr=[yerr_lower, yerr_upper], 
                           fmt='none', color='black', capsize=5)
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Comparison', fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'model_comparison.png')
            fig.savefig(self.save_dir / 'model_comparison.pdf')
        
        return fig
    
    def plot_predictions_vs_actual(
        self,
        predictions: Dict[str, np.ndarray],
        actual: np.ndarray,
        dates: Optional[np.ndarray] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Plot predictions vs actual values.
        
        Args:
            predictions: Dictionary of model names to prediction arrays
            actual: Actual values
            dates: Optional date array
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        x = dates if dates is not None else range(len(actual))
        
        # Full time series
        ax = axes[0]
        ax.plot(x, actual, 'k-', label='Actual', linewidth=2, alpha=0.8)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        for (name, pred), color in zip(predictions.items(), colors):
            ax.plot(x, pred, label=name, color=color, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Date' if dates is not None else 'Time Step')
        ax.set_ylabel('Gold Price')
        ax.set_title('Predictions vs Actual Gold Prices')
        ax.legend()
        
        # Zoomed view (last 100 points)
        ax = axes[1]
        zoom_start = max(0, len(actual) - 100)
        x_zoom = x[zoom_start:]
        
        ax.plot(x_zoom, actual[zoom_start:], 'k-', label='Actual', linewidth=2, alpha=0.8)
        for (name, pred), color in zip(predictions.items(), colors):
            ax.plot(x_zoom, pred[zoom_start:], label=name, color=color, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Date' if dates is not None else 'Time Step')
        ax.set_ylabel('Gold Price')
        ax.set_title('Predictions vs Actual (Last 100 Days)')
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'predictions_vs_actual.png')
            fig.savefig(self.save_dir / 'predictions_vs_actual.pdf')
        
        return fig
    
    def plot_backtest_results(
        self,
        backtest_results: Dict[str, 'BacktestResults'],
        save: bool = True
    ) -> plt.Figure:
        """
        Plot trading backtest results.
        
        Args:
            backtest_results: Dictionary of model names to backtest results
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(backtest_results)))
        
        # Cumulative returns
        ax = axes[0, 0]
        for (name, results), color in zip(backtest_results.items(), colors):
            if hasattr(results, 'cumulative_returns'):
                ax.plot(results.cumulative_returns * 100, label=name, color=color, linewidth=2)
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Cumulative Returns')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Sharpe ratio comparison
        ax = axes[0, 1]
        models = list(backtest_results.keys())
        sharpe_ratios = [r.sharpe_ratio for r in backtest_results.values()]
        ax.bar(models, sharpe_ratios, color=colors)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Max drawdown comparison
        ax = axes[1, 0]
        max_drawdowns = [r.max_drawdown * 100 for r in backtest_results.values()]
        ax.bar(models, max_drawdowns, color=colors)
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown')
        ax.tick_params(axis='x', rotation=45)
        
        # Summary metrics table
        ax = axes[1, 1]
        ax.axis('off')
        
        table_data = []
        for name, results in backtest_results.items():
            table_data.append([
                name,
                f'{results.total_return * 100:.2f}%',
                f'{results.sharpe_ratio:.3f}',
                f'{results.max_drawdown * 100:.2f}%',
                f'{results.win_rate * 100:.1f}%'
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Model', 'Return', 'Sharpe', 'Max DD', 'Win Rate'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle('Trading Strategy Backtest Results', fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.save_dir / 'backtest_results.png')
            fig.savefig(self.save_dir / 'backtest_results.pdf')
        
        return fig


def create_publication_figures(
    training_history: Dict[str, Dict],
    xai_results: Dict[str, Any],
    model_results: Dict[str, Dict],
    backtest_results: Optional[Dict] = None,
    save_dir: str = "figures"
) -> None:
    """
    Create all publication-quality figures.
    
    Args:
        training_history: Training histories for all models
        xai_results: XAI analysis results
        model_results: Model comparison results
        backtest_results: Optional backtest results
        save_dir: Directory to save figures
    """
    logger.info("Creating publication figures...")
    
    # Training visualizations
    train_viz = TrainingVisualizer(save_dir)
    for name, history in training_history.items():
        train_viz.plot_training_curves(history, name)
    train_viz.plot_convergence_comparison(training_history)
    
    # XAI visualizations
    xai_viz = XAIVisualizer(save_dir)
    if 'feature_importance' in xai_results:
        xai_viz.plot_feature_importance(xai_results['feature_importance'])
    if 'shap_values' in xai_results:
        xai_viz.plot_temporal_importance_heatmap(
            xai_results['shap_values'].mean(axis=0),
            xai_results.get('feature_names', [])
        )
    if 'stability_metrics' in xai_results:
        xai_viz.plot_stability_analysis(xai_results['stability_metrics'])
    
    # Results visualizations
    results_viz = ResultsVisualizer(save_dir)
    results_viz.plot_model_comparison(model_results)
    
    if backtest_results:
        results_viz.plot_backtest_results(backtest_results)
    
    logger.info(f"All figures saved to {save_dir}")
