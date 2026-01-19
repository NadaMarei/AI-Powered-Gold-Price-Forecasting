"""
Ablation Study for Gold Price Forecasting Model
================================================

Systematically evaluates the contribution of each architectural component:
1. Volatility-adaptive gating
2. Skip connections
3. Temporal attention
4. Feature transformation layer

Usage:
    python experiments/ablation_study.py --config config/default_config.yaml
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import GoldPriceDataLoader, FeatureEngineer, DataNormalizer, create_data_loaders
from src.models import GoldPriceForecastingModel, HuberDirectionalLoss
from src.training import Trainer, TrainingConfig
from src.evaluation import compute_all_metrics
from src.utils import set_all_seeds, ExperimentReporter, save_results_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_ablation_configs() -> Dict[str, Dict]:
    """Create configurations for ablation study."""
    
    # Full model configuration
    base_config = {
        'use_attention': True,
        'use_skip_connections': True,
        'use_volatility_gating': True,
        'use_feature_transform': True
    }
    
    ablation_configs = {
        'Full Model': base_config.copy(),
        'No Volatility Gating': {**base_config, 'use_volatility_gating': False},
        'No Skip Connections': {**base_config, 'use_skip_connections': False},
        'No Attention': {**base_config, 'use_attention': False},
        'No Feature Transform': {**base_config, 'use_feature_transform': False},
        'Minimal (GRU only)': {
            'use_attention': False,
            'use_skip_connections': False,
            'use_volatility_gating': False,
            'use_feature_transform': False
        }
    }
    
    return ablation_configs


def run_ablation_experiment(
    ablation_name: str,
    ablation_config: Dict,
    train_loader,
    val_loader,
    test_loader,
    full_config: Dict,
    feature_info: Dict,
    device: str,
    seed: int = 42
) -> Dict:
    """Run a single ablation experiment."""
    
    logger.info(f"Running ablation: {ablation_name}")
    set_all_seeds(seed)
    
    # Create model with ablation configuration
    model = GoldPriceForecastingModel(
        num_features=feature_info['n_features'],
        hidden_size=full_config['model']['gru']['hidden_size'],
        num_layers=full_config['model']['gru']['num_layers'],
        num_attention_heads=full_config['model']['gru'].get('attention_heads', 4),
        dropout=full_config['model']['gru']['recurrent_dropout'],
        **ablation_config
    )
    
    n_params = model.count_parameters()
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss function
    loss_fn = HuberDirectionalLoss(
        delta=full_config['training']['loss']['huber_delta'],
        directional_weight=full_config['training']['loss']['directional_weight']
    )
    
    # Training configuration
    train_config = TrainingConfig(
        epochs=full_config['training']['epochs'],
        batch_size=full_config['training']['batch_size'],
        gradient_accumulation_steps=full_config['training']['gradient_accumulation_steps'],
        learning_rate=full_config['training']['learning_rate'],
        weight_decay=full_config['training']['weight_decay'],
        early_stopping_patience=full_config['training']['early_stopping']['patience'],
        seed=seed
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        mlflow_tracking=False
    )
    
    training_results = trainer.train()
    
    # Evaluate on test set
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_previous = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(sequences)
            predictions = outputs['predictions']
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_previous.extend(sequences[:, -1, 0].cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    previous = np.array(all_previous)
    
    metrics = compute_all_metrics(predictions, targets, previous)
    
    return {
        'ablation_name': ablation_name,
        'ablation_config': ablation_config,
        'n_params': n_params,
        'metrics': metrics.to_dict(),
        'training_results': training_results
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("="*80)
    logger.info("ABLATION STUDY")
    logger.info("="*80)
    
    # Prepare data
    loader = GoldPriceDataLoader(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    raw_data = loader.fetch_data()
    
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(raw_data)
    
    feature_columns = [col for col in featured_data.columns 
                       if col not in ['Open', 'High', 'Low', 'Volume']]
    
    normalizer = DataNormalizer(method='standard')
    normalized_data = normalizer.fit_transform(featured_data, feature_columns)
    
    train_loader, val_loader, test_loader, split_info = create_data_loaders(
        normalized_data,
        feature_columns=feature_columns,
        target_column='Close',
        sequence_length=config['data']['sequence_length'],
        batch_size=config['training']['batch_size']
    )
    
    feature_info = {
        'feature_columns': feature_columns,
        'n_features': len(feature_columns)
    }
    
    # Get ablation configurations
    ablation_configs = create_ablation_configs()
    
    # Run ablation experiments
    all_results = {}
    
    for ablation_name, ablation_config in ablation_configs.items():
        seed_results = []
        
        for seed in args.seeds:
            result = run_ablation_experiment(
                ablation_name=ablation_name,
                ablation_config=ablation_config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                full_config=config,
                feature_info=feature_info,
                device=device,
                seed=seed
            )
            seed_results.append(result)
        
        # Aggregate across seeds
        aggregated_metrics = {}
        for key in seed_results[0]['metrics'].keys():
            values = [r['metrics'][key] for r in seed_results if isinstance(r['metrics'][key], (int, float))]
            if values:
                aggregated_metrics[key] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)
        
        all_results[ablation_name] = {
            'ablation_config': ablation_config,
            'n_params': seed_results[0]['n_params'],
            'metrics': aggregated_metrics
        }
        
        logger.info(f"{ablation_name}: RMSE={aggregated_metrics['rmse']:.6f} ± {aggregated_metrics.get('rmse_std', 0):.6f}")
    
    # Generate report
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*80)
    
    # Print comparison table
    print("\n" + "-"*100)
    print(f"{'Configuration':<30} {'RMSE':<15} {'MAE':<15} {'Dir. Acc.':<15} {'Params':<15}")
    print("-"*100)
    
    full_model_rmse = all_results['Full Model']['metrics']['rmse']
    
    for name, results in all_results.items():
        metrics = results['metrics']
        delta_rmse = metrics['rmse'] - full_model_rmse
        print(f"{name:<30} {metrics['rmse']:.6f} {metrics['mae']:.6f} {metrics['directional_accuracy']:.4f} {results['n_params']:,}")
        if name != 'Full Model':
            print(f"{'  (Δ RMSE: ' + f'{delta_rmse:+.6f})':<30}")
    
    print("-"*100)
    
    # Save results
    save_results_summary(all_results, str(output_dir / 'ablation_results.json'))
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Ablation Study Results}")
    latex_lines.append("\\label{tab:ablation}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Configuration & RMSE & MAE & $\\Delta$ RMSE & Params \\\\")
    latex_lines.append("\\midrule")
    
    for name, results in all_results.items():
        metrics = results['metrics']
        delta_rmse = metrics['rmse'] - full_model_rmse
        latex_lines.append(
            f"{name} & {metrics['rmse']:.4f} & {metrics['mae']:.4f} & "
            f"{delta_rmse:+.4f} & {results['n_params']:,} \\\\"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write("\n".join(latex_lines))
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
