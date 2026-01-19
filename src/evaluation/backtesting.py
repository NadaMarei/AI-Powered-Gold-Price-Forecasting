"""
Trading Strategy Backtesting
============================

Implements realistic backtesting framework for evaluating
economic utility of forecasting models.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Container for backtest results."""
    
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading statistics
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Comparison metrics
    excess_return_vs_benchmark: float = 0.0
    information_ratio: float = 0.0
    
    # Time series
    portfolio_values: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'n_trades': self.n_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'excess_return_vs_benchmark': self.excess_return_vs_benchmark,
            'information_ratio': self.information_ratio
        }


class TradingBacktester:
    """
    Realistic trading strategy backtester.
    
    Implements a simple prediction-based trading strategy with
    realistic transaction costs and position sizing.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        position_size: float = 0.1,
        max_position: float = 1.0,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per transaction (as fraction)
            position_size: Base position size (as fraction of capital)
            max_position: Maximum position size
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.max_position = max_position
        self.risk_free_rate = risk_free_rate
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        previous_prices: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None
    ) -> BacktestResults:
        """
        Run backtest on predictions.
        
        Strategy:
        - If predicted return > 0: Long position
        - If predicted return < 0: Short position (or no position)
        - Position size scaled by confidence if provided
        
        Args:
            predictions: Predicted prices
            actual_prices: Actual prices
            previous_prices: Previous day prices (for return calculation)
            confidence: Optional prediction confidence for position sizing
            
        Returns:
            BacktestResults object
        """
        n_days = len(predictions)
        
        if previous_prices is None:
            previous_prices = np.roll(actual_prices, 1)
            previous_prices[0] = actual_prices[0]
        
        # Initialize tracking arrays
        portfolio_values = np.zeros(n_days)
        positions = np.zeros(n_days)
        daily_returns = np.zeros(n_days)
        
        capital = self.initial_capital
        current_position = 0.0
        
        trades = []
        
        for t in range(n_days):
            # Predicted direction
            pred_return = (predictions[t] - previous_prices[t]) / previous_prices[t]
            
            # Determine target position
            if confidence is not None:
                conf = np.clip(confidence[t], 0, 1)
                target_size = self.position_size * conf
            else:
                target_size = self.position_size
            
            if pred_return > 0.001:  # Predict price increase
                target_position = min(target_size, self.max_position)
            elif pred_return < -0.001:  # Predict price decrease
                target_position = -min(target_size, self.max_position)
            else:  # No strong prediction
                target_position = 0.0
            
            # Execute trade if position changes
            position_change = target_position - current_position
            if abs(position_change) > 0.01:
                # Transaction cost
                trade_cost = abs(position_change) * capital * self.transaction_cost
                capital -= trade_cost
                
                trades.append({
                    'day': t,
                    'position_change': position_change,
                    'price': actual_prices[t],
                    'cost': trade_cost
                })
            
            # Calculate return
            actual_return = (actual_prices[t] - previous_prices[t]) / previous_prices[t]
            position_return = current_position * actual_return
            capital *= (1 + position_return)
            
            # Update tracking
            portfolio_values[t] = capital
            positions[t] = current_position
            daily_returns[t] = position_return
            
            # Update position
            current_position = target_position
        
        # Calculate metrics
        results = self._calculate_metrics(
            portfolio_values, daily_returns, positions, trades, actual_prices
        )
        
        return results
    
    def _calculate_metrics(
        self,
        portfolio_values: np.ndarray,
        daily_returns: np.ndarray,
        positions: np.ndarray,
        trades: List[Dict],
        actual_prices: np.ndarray
    ) -> BacktestResults:
        """Calculate all backtest metrics."""
        
        # Returns
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        cumulative_returns = portfolio_values / self.initial_capital - 1
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = daily_returns - self.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # Sortino ratio (using only negative returns for denominator)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trading statistics
        n_trades = len(trades)
        
        if n_trades > 0:
            # Calculate trade returns
            trade_returns = []
            for i, trade in enumerate(trades):
                if i < len(trades) - 1:
                    entry_price = actual_prices[trade['day']]
                    exit_price = actual_prices[trades[i + 1]['day']]
                    direction = np.sign(trade['position_change'])
                    ret = direction * (exit_price - entry_price) / entry_price
                    trade_returns.append(ret)
            
            if trade_returns:
                trade_returns = np.array(trade_returns)
                wins = trade_returns[trade_returns > 0]
                losses = trade_returns[trade_returns < 0]
                
                win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
                avg_win = np.mean(wins) if len(wins) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                total_wins = np.sum(wins) if len(wins) > 0 else 0
                total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Benchmark comparison (buy and hold)
        benchmark_return = (actual_prices[-1] / actual_prices[0]) - 1
        excess_return_vs_benchmark = total_return - benchmark_return
        
        # Information ratio
        benchmark_daily_returns = np.diff(actual_prices) / actual_prices[:-1]
        if len(benchmark_daily_returns) == len(daily_returns) - 1:
            benchmark_daily_returns = np.concatenate([[0], benchmark_daily_returns])
        tracking_error = np.std(daily_returns - benchmark_daily_returns[:len(daily_returns)])
        information_ratio = excess_return_vs_benchmark / tracking_error / np.sqrt(n_years) if tracking_error > 0 else 0
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_returns=cumulative_returns,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            n_trades=n_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            excess_return_vs_benchmark=excess_return_vs_benchmark,
            information_ratio=information_ratio,
            portfolio_values=portfolio_values,
            positions=positions
        )
    
    def run_monte_carlo_backtest(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        n_simulations: int = 1000,
        noise_std: float = 0.01
    ) -> Dict[str, any]:
        """
        Run Monte Carlo simulation for robustness analysis.
        
        Args:
            predictions: Predicted prices
            actual_prices: Actual prices
            n_simulations: Number of simulations
            noise_std: Standard deviation of prediction noise
            
        Returns:
            Dictionary with simulation results
        """
        results = []
        
        for _ in range(n_simulations):
            # Add noise to predictions
            noisy_predictions = predictions * (1 + np.random.normal(0, noise_std, len(predictions)))
            
            # Run backtest
            result = self.run_backtest(noisy_predictions, actual_prices)
            results.append(result.to_dict())
        
        # Aggregate results
        metrics = list(results[0].keys())
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if isinstance(r[metric], (int, float))]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_5pct'] = np.percentile(values, 5)
                aggregated[f'{metric}_95pct'] = np.percentile(values, 95)
        
        return aggregated
