"""
Main Experiment Runner for Gold Price Forecasting
==================================================

This script runs the complete experimental pipeline:
1. Data loading and preprocessing
2. Training all models (Proposed GRU, LSTM, SARIMA, Gradient Boosting)
3. Evaluation with statistical tests
4. XAI analysis
5. Trading backtest
6. Report generation

Usage:
    python experiments/run_experiment.py --config config/default_config.yaml
    python experiments/run_experiment.py --seeds 42 123 456 789 1011
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import GoldPriceDataLoader, FeatureEngineer, DataNormalizer, create_data_loaders
from src.models import (
    GoldPriceForecastingModel, 
    LSTMBaseline, 
    SARIMAModel, 
    GradientBoostingModel,
    HuberDirectionalLoss
)
from src.training import Trainer, TrainingConfig
from src.evaluation import (
    compute_all_metrics, 
    diebold_mariano_test,
    TradingBacktester
)
from src.xai import SHAPExplainer, compute_stability_metrics, generate_economic_scenarios
from src.utils import (
    set_all_seeds, 
    save_experiment_config,
    TrainingVisualizer, 
    XAIVisualizer, 
    ResultsVisualizer,
    ExperimentReporter,
    save_results_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict) -> tuple:
    """
    Load and preprocess data.
    
    Returns:
        Tuple of (data_loaders, feature_info, raw_data)
    """
    logger.info("Loading and preprocessing data...")
    
    # Load data
    loader = GoldPriceDataLoader(
        ticker=config['data']['ticker'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    raw_data = loader.fetch_data()
    
    # Feature engineering
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(raw_data)
    
    # Select features
    feature_columns = [col for col in featured_data.columns 
                       if col not in ['Open', 'High', 'Low', 'Volume']]
    
    # Normalize
    normalizer = DataNormalizer(method='standard')
    normalized_data = normalizer.fit_transform(featured_data, feature_columns)
    
    # Create data loaders
    train_loader, val_loader, test_loader, split_info = create_data_loaders(
        normalized_data,
        feature_columns=feature_columns,
        target_column='Close',
        sequence_length=config['data']['sequence_length'],
        forecast_horizon=config['data']['forecast_horizon'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        batch_size=config['training']['batch_size'],
        volatility_column='Volatility_20' if 'Volatility_20' in feature_columns else None
    )
    
    feature_info = {
        'feature_columns': feature_columns,
        'normalizer': normalizer,
        'n_features': len(feature_columns),
        'split_info': split_info
    }
    
    logger.info(f"Data prepared: {split_info}")
    
    return (train_loader, val_loader, test_loader), feature_info, featured_data


def train_proposed_model(
    train_loader,
    val_loader,
    config: Dict,
    feature_info: Dict,
    seed: int = 42
) -> tuple:
    """Train the proposed GRU model."""
    logger.info("Training Proposed GRU Model...")
    set_all_seeds(seed)
    
    model_config = config['model']['gru']
    n_features = feature_info['n_features']
    
    # Create model
    model = GoldPriceForecastingModel(
        num_features=n_features,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_attention_heads=model_config.get('attention_heads', 4),
        dropout=model_config['recurrent_dropout'],
        use_attention=model_config['use_attention'],
        use_skip_connections=model_config['use_skip_connections'],
        use_volatility_gating=model_config['use_volatility_gating'],
        use_feature_transform=model_config.get('feature_transform_layers', True)
    )
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Loss function
    loss_config = config['training']['loss']
    loss_fn = HuberDirectionalLoss(
        delta=loss_config['huber_delta'],
        directional_weight=loss_config['directional_weight']
    )
    
    # Training configuration
    train_config = TrainingConfig(
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer_type=config['training']['optimizer'],
        scheduler_type=config['training']['scheduler'],
        warmup_epochs=config['training']['warmup_epochs'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        gradient_clip=config['training']['regularization']['gradient_clip'],
        seed=seed
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        mlflow_tracking=True
    )
    
    results = trainer.train()
    
    return model, results


def train_lstm_baseline(
    train_loader,
    val_loader,
    config: Dict,
    feature_info: Dict,
    seed: int = 42
) -> tuple:
    """Train LSTM baseline model."""
    logger.info("Training LSTM Baseline...")
    set_all_seeds(seed)
    
    model_config = config['model']['lstm_baseline']
    n_features = feature_info['n_features']
    
    model = LSTMBaseline(
        num_features=n_features,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    loss_fn = torch.nn.MSELoss()
    
    train_config = TrainingConfig(
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
        seed=seed
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=train_config,
        mlflow_tracking=False
    )
    
    results = trainer.train()
    
    return model, results


def train_sarima(data: np.ndarray, config: Dict) -> SARIMAModel:
    """Train SARIMA model."""
    logger.info("Training SARIMA Model...")
    
    sarima_config = config['benchmarks']['sarima']
    
    model = SARIMAModel(
        seasonal_period=sarima_config['seasonal_period'],
        max_p=sarima_config['max_p'],
        max_d=sarima_config['max_d'],
        max_q=sarima_config['max_q'],
        auto_select=sarima_config['auto_select']
    )
    
    model.fit(data)
    
    return model


def train_gradient_boosting(data: np.ndarray, config: Dict) -> GradientBoostingModel:
    """Train Gradient Boosting model."""
    logger.info("Training Gradient Boosting Model...")
    
    gb_config = config['benchmarks']['gradient_boosting']
    
    model = GradientBoostingModel(
        n_estimators=gb_config['n_estimators'],
        max_depth=gb_config['max_depth'],
        learning_rate=gb_config['learning_rate'],
        subsample=gb_config['subsample']
    )
    
    model.fit(data)
    
    return model


def evaluate_model(
    model,
    test_loader,
    model_type: str = 'pytorch',
    device: str = 'cpu'
) -> Dict:
    """Evaluate a model on test data."""
    logger.info(f"Evaluating {model_type} model...")
    
    all_predictions = []
    all_targets = []
    all_previous = []
    
    if model_type == 'pytorch':
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(sequences) if model_type == 'lstm' else model(sequences)
                predictions = outputs['predictions']
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_previous.extend(sequences[:, -1, 0].cpu().numpy())
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    previous = np.array(all_previous)
    
    # Compute metrics
    metrics = compute_all_metrics(predictions, targets, previous)
    
    return {
        'metrics': metrics.to_dict(),
        'predictions': predictions,
        'targets': targets
    }


def run_xai_analysis(
    model,
    test_loader,
    feature_names: List[str],
    config: Dict,
    device: str = 'cpu'
) -> Dict:
    """Run XAI analysis on the model."""
    logger.info("Running XAI Analysis...")
    
    # Prepare background data
    background_data = []
    for batch in test_loader:
        background_data.append(batch['sequences'])
        if len(background_data) * batch['sequences'].shape[0] >= config['xai']['shap']['background_samples']:
            break
    
    background_data = torch.cat(background_data, dim=0)[:config['xai']['shap']['background_samples']]
    
    # Create explainer
    explainer = SHAPExplainer(
        model=model,
        background_data=background_data,
        feature_names=feature_names,
        device=device
    )
    
    # Get test samples for explanation
    test_samples = []
    for batch in test_loader:
        test_samples.append(batch['sequences'])
        if len(test_samples) * batch['sequences'].shape[0] >= config['xai']['shap']['test_samples']:
            break
    
    test_samples = torch.cat(test_samples, dim=0)[:config['xai']['shap']['test_samples']]
    
    # Compute SHAP values
    explanation = explainer.explain(test_samples)
    
    # Stability analysis
    stability_metrics = compute_stability_metrics(
        model=model,
        explainer=explainer,
        X=test_samples,
        feature_names=feature_names
    )
    
    # Economic scenario analysis
    scenario_results = generate_economic_scenarios(
        model=model,
        x=test_samples[0],
        feature_names=feature_names,
        device=device
    )
    
    return {
        'shap_values': explanation['shap_values'],
        'feature_importance': explanation['feature_importance'],
        'temporal_importance': explanation['temporal_importance'],
        'stability_metrics': stability_metrics,
        'scenario_analysis': scenario_results,
        'feature_names': feature_names
    }


def run_backtest(
    predictions: Dict[str, np.ndarray],
    actual_prices: np.ndarray,
    config: Dict
) -> Dict:
    """Run trading backtest for all models."""
    logger.info("Running Trading Backtest...")
    
    backtest_config = config['evaluation']['trading_backtest']
    
    backtester = TradingBacktester(
        initial_capital=backtest_config['initial_capital'],
        transaction_cost=backtest_config['transaction_cost'],
        position_size=backtest_config['position_size']
    )
    
    results = {}
    for model_name, preds in predictions.items():
        bt_result = backtester.run_backtest(preds, actual_prices)
        results[model_name] = bt_result
        logger.info(f"{model_name}: Sharpe={bt_result.sharpe_ratio:.3f}, Return={bt_result.total_return*100:.2f}%")
    
    return results


def run_statistical_tests(
    results: Dict[str, Dict]
) -> Dict:
    """Run statistical significance tests."""
    logger.info("Running Statistical Tests...")
    
    # Diebold-Mariano tests
    dm_results = {}
    
    model_names = list(results.keys())
    baseline = 'LSTM Baseline'
    
    if baseline in results:
        baseline_errors = results[baseline]['targets'] - results[baseline]['predictions']
        
        for model_name in model_names:
            if model_name != baseline:
                model_errors = results[model_name]['targets'] - results[model_name]['predictions']
                
                dm_test = diebold_mariano_test(
                    errors_1=model_errors,
                    errors_2=baseline_errors
                )
                dm_results[f'{model_name} vs {baseline}'] = dm_test
    
    return dm_results


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Gold Price Forecasting Experiment')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds for multiple runs')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--skip_sarima', action='store_true',
                       help='Skip SARIMA model (slow)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    exp_dir = save_experiment_config(config, str(output_dir), 'gold_forecasting')
    
    logger.info("="*80)
    logger.info("GOLD PRICE FORECASTING EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {exp_dir}")
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Prepare data
    data_loaders, feature_info, raw_data = prepare_data(config)
    train_loader, val_loader, test_loader = data_loaders
    
    # Store all results across seeds
    all_results = {seed: {} for seed in args.seeds}
    
    for seed in args.seeds:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running with seed: {seed}")
        logger.info(f"{'='*40}")
        
        # Train Proposed Model
        proposed_model, proposed_history = train_proposed_model(
            train_loader, val_loader, config, feature_info, seed
        )
        proposed_results = evaluate_model(proposed_model, test_loader, 'pytorch', device)
        all_results[seed]['Proposed Model'] = {
            'metrics': proposed_results['metrics'],
            'history': proposed_history,
            'predictions': proposed_results['predictions'],
            'targets': proposed_results['targets']
        }
        
        # Train LSTM Baseline
        lstm_model, lstm_history = train_lstm_baseline(
            train_loader, val_loader, config, feature_info, seed
        )
        lstm_results = evaluate_model(lstm_model, test_loader, 'pytorch', device)
        all_results[seed]['LSTM Baseline'] = {
            'metrics': lstm_results['metrics'],
            'history': lstm_history,
            'predictions': lstm_results['predictions'],
            'targets': lstm_results['targets']
        }
        
        # Statistical models (only first seed)
        if seed == args.seeds[0]:
            # SARIMA
            if not args.skip_sarima:
                try:
                    close_prices = raw_data['Close'].values
                    train_size = int(len(close_prices) * config['data']['train_ratio'])
                    sarima_model = train_sarima(close_prices[:train_size], config)
                    
                    # Generate predictions
                    sarima_preds = []
                    for i in range(train_size, len(close_prices)):
                        sarima_model.fit(close_prices[:i])
                        pred = sarima_model.predict(steps=1)[0]
                        sarima_preds.append(pred)
                    
                    sarima_metrics = compute_all_metrics(
                        np.array(sarima_preds),
                        close_prices[train_size:len(sarima_preds)+train_size]
                    )
                    all_results[seed]['SARIMA'] = {'metrics': sarima_metrics.to_dict()}
                except Exception as e:
                    logger.warning(f"SARIMA failed: {e}")
            
            # Gradient Boosting
            try:
                close_prices = raw_data['Close'].values
                train_size = int(len(close_prices) * config['data']['train_ratio'])
                gb_model = train_gradient_boosting(close_prices[:train_size], config)
                
                # Generate predictions
                gb_preds = []
                seq_len = config['data']['sequence_length']
                for i in range(train_size, len(close_prices)):
                    hist = close_prices[max(0, i-seq_len):i]
                    pred = gb_model.predict(hist)
                    gb_preds.append(pred)
                
                gb_metrics = compute_all_metrics(
                    np.array(gb_preds),
                    close_prices[train_size:len(gb_preds)+train_size]
                )
                all_results[seed]['Gradient Boosting'] = {'metrics': gb_metrics.to_dict()}
            except Exception as e:
                logger.warning(f"Gradient Boosting failed: {e}")
    
    # Aggregate results across seeds
    logger.info("\n" + "="*80)
    logger.info("AGGREGATING RESULTS")
    logger.info("="*80)
    
    aggregated_results = {}
    for model_name in all_results[args.seeds[0]].keys():
        metrics_list = [
            all_results[seed][model_name]['metrics']
            for seed in args.seeds
            if model_name in all_results[seed]
        ]
        
        if metrics_list:
            aggregated = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if isinstance(m[key], (int, float))]
                if values:
                    aggregated[key] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
            
            aggregated_results[model_name] = {'metrics': aggregated}
    
    # Run XAI Analysis (on best model)
    logger.info("\n" + "="*80)
    logger.info("XAI ANALYSIS")
    logger.info("="*80)
    
    xai_results = run_xai_analysis(
        model=proposed_model,
        test_loader=test_loader,
        feature_names=feature_info['feature_columns'],
        config=config,
        device=device
    )
    
    # Statistical tests
    dm_tests = run_statistical_tests(all_results[args.seeds[0]])
    
    # Trading backtest
    predictions_dict = {
        name: result['predictions']
        for name, result in all_results[args.seeds[0]].items()
        if 'predictions' in result
    }
    
    if predictions_dict:
        backtest_results = run_backtest(
            predictions_dict,
            all_results[args.seeds[0]]['Proposed Model']['targets'],
            config
        )
    else:
        backtest_results = {}
    
    # Generate visualizations
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Training curves
    train_viz = TrainingVisualizer(str(figures_dir))
    for model_name, results in all_results[args.seeds[0]].items():
        if 'history' in results:
            train_viz.plot_training_curves(results['history'], model_name)
    
    # XAI visualizations
    xai_viz = XAIVisualizer(str(figures_dir))
    xai_viz.plot_feature_importance(xai_results['feature_importance'])
    xai_viz.plot_stability_analysis(xai_results['stability_metrics'])
    
    # Results visualizations
    results_viz = ResultsVisualizer(str(figures_dir))
    results_viz.plot_model_comparison(aggregated_results)
    
    if backtest_results:
        results_viz.plot_backtest_results(backtest_results)
    
    # Generate report
    logger.info("\n" + "="*80)
    logger.info("GENERATING REPORT")
    logger.info("="*80)
    
    reporter = ExperimentReporter('gold_forecasting', str(output_dir / 'reports'))
    for model_name, results in aggregated_results.items():
        reporter.add_model_results(model_name, results)
    
    reporter.add_metadata('seeds', args.seeds)
    reporter.add_metadata('dm_tests', dm_tests)
    report = reporter.generate_report()
    
    # Save all results
    save_results_summary({
        'aggregated_results': aggregated_results,
        'xai_results': {
            'feature_importance': xai_results['feature_importance'],
            'stability_metrics': xai_results['stability_metrics']
        },
        'dm_tests': dm_tests,
        'backtest_results': {k: v.to_dict() for k, v in backtest_results.items()} if backtest_results else {}
    }, str(output_dir / 'results_summary.json'))
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    print(report['summary'])
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Experiment completed successfully!")


if __name__ == '__main__':
    main()
