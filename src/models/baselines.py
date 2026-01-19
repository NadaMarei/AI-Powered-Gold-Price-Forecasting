"""
Baseline Models for Gold Price Forecasting
==========================================

Implements benchmark models for comparison:
1. LSTM Baseline - Standard deep learning baseline
2. SARIMA - Statistical time series model
3. Gradient Boosting - Traditional ML with temporal features
"""

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class LSTMBaseline(nn.Module):
    """
    Standard LSTM baseline for comparison.
    
    Implements a vanilla LSTM architecture with comparable
    parameters to the proposed GRU model.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM baseline.
        
        Args:
            num_features: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output network
        lstm_output_size = hidden_size * self.num_directions
        self.output_network = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequences [batch, seq_len, num_features]
            
        Returns:
            Dictionary with predictions
        """
        # LSTM forward
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_output[:, -1, :]
        
        # Generate prediction
        predictions = self.output_network(last_output)
        
        return {
            'predictions': predictions,
            'hidden_states': hidden,
            'lstm_output': lstm_output
        }
    
    def predict(self, x: Tensor) -> Tensor:
        """Generate predictions."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output['predictions']
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SARIMAModel:
    """
    SARIMA model for statistical baseline.
    
    Uses automatic parameter selection via AIC/BIC criteria
    with seasonal components for trading week patterns.
    """
    
    def __init__(
        self,
        seasonal_period: int = 5,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        auto_select: bool = True
    ):
        """
        Initialize SARIMA model.
        
        Args:
            seasonal_period: Seasonal period (5 for trading week)
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            auto_select: Whether to automatically select parameters
        """
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.auto_select = auto_select
        self.model = None
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None
        
    def fit(self, y: np.ndarray) -> 'SARIMAModel':
        """
        Fit SARIMA model to training data.
        
        Args:
            y: Training time series
            
        Returns:
            Self for method chaining
        """
        try:
            import pmdarima as pm
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            if self.auto_select:
                # Auto-select parameters using pmdarima
                logger.info("Auto-selecting SARIMA parameters...")
                auto_model = pm.auto_arima(
                    y,
                    start_p=1, start_q=1,
                    max_p=self.max_p, max_q=self.max_q,
                    max_d=self.max_d,
                    seasonal=True, m=self.seasonal_period,
                    start_P=0, start_Q=0,
                    max_P=2, max_Q=2,
                    d=None, D=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    n_jobs=-1
                )
                self.order = auto_model.order
                self.seasonal_order = auto_model.seasonal_order
                logger.info(f"Selected order: {self.order}, seasonal: {self.seasonal_order}")
            else:
                self.order = (1, 1, 1)
                self.seasonal_order = (1, 1, 1, self.seasonal_period)
            
            # Fit final model using statsmodels
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted_model = self.model.fit(disp=False)
            
            logger.info(f"SARIMA model fitted. AIC: {self.fitted_model.aic:.2f}")
            
        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {e}")
            # Fall back to simple model
            self.order = (1, 0, 0)
            self.seasonal_order = (0, 0, 0, 0)
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(y, order=self.order)
            self.fitted_model = self.model.fit(disp=False)
        
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def predict_in_sample(self) -> np.ndarray:
        """Get in-sample predictions."""
        if self.fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.fitted_model.fittedvalues
    
    def get_model_summary(self) -> Dict:
        """Get model summary statistics."""
        if self.fitted_model is None:
            return {}
        
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.llf
        }


class GradientBoostingModel:
    """
    Gradient Boosting model with temporal feature engineering.
    
    Creates lagged features and rolling statistics for ML-based
    time series forecasting.
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        n_lags: int = 20,
        n_rolling_windows: List[int] = [5, 10, 20]
    ):
        """
        Initialize Gradient Boosting model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            n_lags: Number of lag features to create
            n_rolling_windows: Window sizes for rolling features
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_lags = n_lags
        self.n_rolling_windows = n_rolling_windows
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=0
        )
        
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def _create_temporal_features(self, y: np.ndarray) -> np.ndarray:
        """
        Create temporal features from time series.
        
        Args:
            y: Time series values
            
        Returns:
            Feature matrix
        """
        df = pd.DataFrame({'value': y})
        
        # Lag features
        for i in range(1, self.n_lags + 1):
            df[f'lag_{i}'] = df['value'].shift(i)
        
        # Rolling statistics
        for window in self.n_rolling_windows:
            df[f'rolling_mean_{window}'] = df['value'].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df['value'].shift(1).rolling(window).std()
            df[f'rolling_min_{window}'] = df['value'].shift(1).rolling(window).min()
            df[f'rolling_max_{window}'] = df['value'].shift(1).rolling(window).max()
        
        # Returns
        df['return_1'] = df['value'].pct_change(1).shift(1)
        df['return_5'] = df['value'].pct_change(5).shift(1)
        
        # Drop original value and rows with NaN
        df = df.drop('value', axis=1)
        
        self.feature_names = df.columns.tolist()
        
        return df.values
    
    def fit(self, y: np.ndarray) -> 'GradientBoostingModel':
        """
        Fit model on training data.
        
        Args:
            y: Training time series
            
        Returns:
            Self for method chaining
        """
        logger.info("Creating temporal features for Gradient Boosting...")
        X = self._create_temporal_features(y)
        
        # Remove rows with NaN
        valid_idx = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_valid)
        
        logger.info(f"Training Gradient Boosting on {len(X_valid)} samples with {X_valid.shape[1]} features...")
        self.model.fit(X_scaled, y_valid)
        
        self.is_fitted = True
        logger.info(f"Gradient Boosting training complete. Best iteration: {self.model.n_estimators_}")
        
        return self
    
    def predict(self, y_history: np.ndarray) -> float:
        """
        Generate single-step prediction.
        
        Args:
            y_history: Historical values for feature creation
            
        Returns:
            Predicted value
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = self._create_temporal_features(y_history)
        
        # Use last valid row
        last_valid_idx = -1
        while np.isnan(X[last_valid_idx]).any():
            last_valid_idx -= 1
        
        X_last = X[last_valid_idx:last_valid_idx + 1]
        X_scaled = self.scaler.transform(X_last)
        
        return self.model.predict(X_scaled)[0]
    
    def predict_batch(self, y_sequences: np.ndarray) -> np.ndarray:
        """
        Generate predictions for multiple sequences.
        
        Args:
            y_sequences: Array of historical sequences [n_samples, seq_len]
            
        Returns:
            Array of predictions
        """
        predictions = []
        for seq in y_sequences:
            pred = self.predict(seq)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances."""
        if not self.is_fitted:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def get_model_summary(self) -> Dict:
        """Get model summary."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_importances': self.get_feature_importance() if self.is_fitted else {}
        }


def create_comparable_lstm(gru_config: Dict) -> LSTMBaseline:
    """
    Create LSTM baseline with comparable parameters to GRU model.
    
    Args:
        gru_config: Configuration dictionary from GRU model
        
    Returns:
        LSTMBaseline instance
    """
    return LSTMBaseline(
        num_features=gru_config.get('num_features', 10),
        hidden_size=gru_config.get('hidden_size', 128),
        num_layers=gru_config.get('num_layers', 3),
        dropout=gru_config.get('dropout', 0.2),
        bidirectional=True
    )
