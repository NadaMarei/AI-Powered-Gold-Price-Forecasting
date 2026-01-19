"""
Reporting Utilities
===================

Generate publication-ready reports and tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ExperimentReporter:
    """
    Generate comprehensive experiment reports.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "reports"):
        """
        Initialize reporter.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for reports
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    def add_model_results(self, model_name: str, results: Dict):
        """Add results for a model."""
        self.results[model_name] = results
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the report."""
        self.metadata[key] = value
    
    def generate_summary(self) -> str:
        """
        Generate text summary of results.
        
        Returns:
            Summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Model comparison
        lines.append("MODEL PERFORMANCE COMPARISON")
        lines.append("-" * 40)
        
        for model_name, results in self.results.items():
            lines.append(f"\n{model_name}:")
            if 'metrics' in results:
                metrics = results['metrics']
                lines.append(f"  RMSE: {metrics.get('rmse', 'N/A'):.6f}")
                lines.append(f"  MAE: {metrics.get('mae', 'N/A'):.6f}")
                lines.append(f"  MAPE: {metrics.get('mape', 'N/A'):.4f}%")
                lines.append(f"  Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.4f}")
        
        # Best model
        lines.append("\n" + "-" * 40)
        if self.results:
            best_model = min(
                self.results.items(),
                key=lambda x: x[1].get('metrics', {}).get('rmse', float('inf'))
            )[0]
            lines.append(f"Best Model (by RMSE): {best_model}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def generate_report(self, include_latex: bool = True) -> Dict[str, str]:
        """
        Generate full report.
        
        Args:
            include_latex: Whether to include LaTeX tables
            
        Returns:
            Dictionary with report sections
        """
        report = {}
        
        # Text summary
        report['summary'] = self.generate_summary()
        
        # JSON results
        report['json'] = json.dumps({
            'metadata': self.metadata,
            'results': self.results
        }, indent=2, default=str)
        
        # LaTeX tables
        if include_latex:
            report['latex_metrics'] = self._generate_latex_metrics_table()
            report['latex_significance'] = self._generate_latex_significance_table()
        
        # Save reports
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write(report['summary'])
        
        with open(self.output_dir / 'results.json', 'w') as f:
            f.write(report['json'])
        
        if include_latex:
            with open(self.output_dir / 'metrics_table.tex', 'w') as f:
                f.write(report['latex_metrics'])
            with open(self.output_dir / 'significance_table.tex', 'w') as f:
                f.write(report['latex_significance'])
        
        logger.info(f"Reports saved to {self.output_dir}")
        
        return report
    
    def _generate_latex_metrics_table(self) -> str:
        """Generate LaTeX table for metrics."""
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Forecast Performance Metrics}")
        lines.append("\\label{tab:metrics}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Model & RMSE & MAE & MAPE (\\%) & Dir. Acc. \\\\")
        lines.append("\\midrule")
        
        for model_name, results in self.results.items():
            metrics = results.get('metrics', {})
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            mape = metrics.get('mape', 0)
            dir_acc = metrics.get('directional_accuracy', 0)
            
            # Format with confidence intervals if available
            rmse_ci = metrics.get('rmse_ci', (0, 0))
            if rmse_ci[0] > 0:
                rmse_str = f"{rmse:.4f} ({rmse_ci[0]:.4f}, {rmse_ci[1]:.4f})"
            else:
                rmse_str = f"{rmse:.4f}"
            
            lines.append(f"{model_name} & {rmse_str} & {mae:.4f} & {mape:.2f} & {dir_acc:.4f} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def _generate_latex_significance_table(self) -> str:
        """Generate LaTeX table for statistical significance."""
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Diebold-Mariano Test Results}")
        lines.append("\\label{tab:significance}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("Comparison & DM Statistic & p-value \\\\")
        lines.append("\\midrule")
        
        # Add DM test results if available
        for model_name, results in self.results.items():
            if 'dm_tests' in results:
                for comparison, dm_result in results['dm_tests'].items():
                    stat = dm_result.get('dm_statistic', 0)
                    pval = dm_result.get('p_value', 1)
                    sig = dm_result.get('significance', '')
                    lines.append(f"{comparison} & {stat:.3f} & {pval:.4f}{sig} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\begin{tablenotes}")
        lines.append("\\small")
        lines.append("\\item Note: * p<0.10, ** p<0.05, *** p<0.01")
        lines.append("\\end{tablenotes}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)


def generate_latex_tables(
    model_results: Dict[str, Dict],
    backtest_results: Optional[Dict] = None,
    output_dir: str = "reports"
) -> Dict[str, str]:
    """
    Generate all LaTeX tables for the paper.
    
    Args:
        model_results: Dictionary of model results
        backtest_results: Optional backtest results
        output_dir: Output directory
        
    Returns:
        Dictionary of table names to LaTeX strings
    """
    tables = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Metrics comparison table
    tables['metrics'] = _generate_metrics_table(model_results)
    
    # Ablation study table
    tables['ablation'] = _generate_ablation_table(model_results)
    
    # Backtest results table
    if backtest_results:
        tables['backtest'] = _generate_backtest_table(backtest_results)
    
    # Save tables
    for name, content in tables.items():
        with open(output_path / f'{name}_table.tex', 'w') as f:
            f.write(content)
    
    return tables


def _generate_metrics_table(results: Dict) -> str:
    """Generate metrics comparison table."""
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of forecasting models on gold price prediction.}")
    lines.append("\\label{tab:model_comparison}")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append("Model & RMSE & MAE & MAPE & Dir. Acc. & R² & Sharpe & Params \\\\")
    lines.append("\\midrule")
    
    for model_name, model_results in results.items():
        metrics = model_results.get('metrics', {})
        lines.append(
            f"{model_name} & "
            f"{metrics.get('rmse', 0):.4f} & "
            f"{metrics.get('mae', 0):.4f} & "
            f"{metrics.get('mape', 0):.2f}\\% & "
            f"{metrics.get('directional_accuracy', 0):.4f} & "
            f"{metrics.get('r_squared', 0):.4f} & "
            f"{metrics.get('sharpe_ratio', 0):.3f} & "
            f"{metrics.get('n_params', 0):,} \\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def _generate_ablation_table(results: Dict) -> str:
    """Generate ablation study table."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study results.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Configuration & RMSE & MAE & $\\Delta$ RMSE \\\\")
    lines.append("\\midrule")
    
    # Full model as baseline
    baseline_rmse = results.get('Proposed Model', {}).get('metrics', {}).get('rmse', 0)
    
    ablation_configs = [
        ('Full Model (Proposed)', 'Proposed Model'),
        ('w/o Volatility Gating', 'No_Volatility_Gating'),
        ('w/o Skip Connections', 'No_Skip_Connections'),
        ('w/o Temporal Attention', 'No_Attention'),
        ('w/o Feature Transform', 'No_Feature_Transform')
    ]
    
    for display_name, key in ablation_configs:
        if key in results:
            metrics = results[key].get('metrics', {})
            rmse = metrics.get('rmse', 0)
            mae = metrics.get('mae', 0)
            delta = rmse - baseline_rmse
            
            lines.append(f"{display_name} & {rmse:.4f} & {mae:.4f} & {delta:+.4f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def _generate_backtest_table(results: Dict) -> str:
    """Generate backtest results table."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Trading strategy backtest results.}")
    lines.append("\\label{tab:backtest}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & Return & Sharpe & Sortino & Max DD & Win Rate & Profit Factor \\\\")
    lines.append("\\midrule")
    
    for model_name, bt_result in results.items():
        lines.append(
            f"{model_name} & "
            f"{bt_result.total_return * 100:.2f}\\% & "
            f"{bt_result.sharpe_ratio:.3f} & "
            f"{bt_result.sortino_ratio:.3f} & "
            f"{bt_result.max_drawdown * 100:.2f}\\% & "
            f"{bt_result.win_rate * 100:.1f}\\% & "
            f"{bt_result.profit_factor:.2f} \\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def save_results_summary(
    results: Dict,
    filepath: str = "results/summary.json"
) -> None:
    """
    Save comprehensive results summary to JSON.
    
    Args:
        results: All experiment results
        filepath: Output filepath
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
