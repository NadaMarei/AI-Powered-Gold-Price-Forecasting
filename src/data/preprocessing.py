"""
Feature Engineering and Preprocessing Module
=============================================

Implements comprehensive feature engineering for financial time series
including technical indicators, volatility measures, and regime detection.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for gold price forecasting.
    
    Implements technical indicators, volatility measures, and market regime
    detection following best practices in financial ML research.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary for feature parameters
        """
        self.config = config or {}
        self.feature_names: List[str] = []
        
    def compute_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from raw OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all computed features
        """
        df = data.copy()
        
        # Price-based features
        df = self._compute_returns(df)
        df = self._compute_volatility_features(df)
        df = self._compute_moving_averages(df)
        df = self._compute_momentum_indicators(df)
        df = self._compute_volatility_bands(df)
        df = self._compute_volume_features(df)
        df = self._compute_regime_features(df)
        df = self._compute_calendar_features(df)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Replace infinity values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN values from feature computation
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values after feature engineering")
        
        return df
    
    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute various return measures."""
        # Simple returns
        df['Returns'] = df['Close'].pct_change()
        
        # Log returns (for volatility modeling)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multi-period returns
        for period in [5, 10, 20]:
            df[f'Returns_{period}d'] = df['Close'].pct_change(period)
        
        # Overnight returns
        df['Overnight_Return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Intraday range
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
        
        return df
    
    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility measures."""
        # Rolling volatility
        for window in [5, 10, 20, 60]:
            df[f'Volatility_{window}'] = df['Log_Returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Parkinson volatility (uses high-low range)
        # Add small epsilon to prevent log(0) and division issues
        high_low_ratio = df['High'] / df['Low'].replace(0, np.nan)
        high_low_ratio = high_low_ratio.clip(lower=1e-10)
        df['Parkinson_Vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(high_low_ratio) ** 2)
        ).rolling(window=20).mean() * np.sqrt(252)
        
        # Garman-Klass volatility
        log_hl = np.log(high_low_ratio) ** 2
        close_open_ratio = df['Close'] / df['Open'].replace(0, np.nan)
        close_open_ratio = close_open_ratio.clip(lower=1e-10)
        log_co = np.log(close_open_ratio) ** 2
        gk_inner = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        gk_inner = gk_inner.clip(lower=0)  # Ensure non-negative before sqrt
        df['GK_Vol'] = np.sqrt(gk_inner).rolling(window=20).mean() * np.sqrt(252)
        
        # Volatility ratio (short-term vs long-term)
        df['Vol_Ratio'] = df['Volatility_10'] / df['Volatility_60'].replace(0, np.nan)
        
        return df
    
    def _compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving averages and related features."""
        # Simple moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_Dist'] = (df['Close'] - df[f'MA_{window}']) / df[f'MA_{window}']
        
        # Exponential moving averages
        for span in [12, 26, 50]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        
        # MA crossover signals
        df['MA_Cross_10_50'] = (df['MA_10'] > df['MA_50']).astype(int)
        df['MA_Cross_50_200'] = (df['MA_50'] > df['MA_200']).astype(int)
        
        return df
    
    def _compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum and oscillator indicators."""
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        period = 14
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        high_low_range = (high_max - low_min).replace(0, np.nan)
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / high_low_range
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
        
        # Williams %R
        df['Williams_R'] = -100 * (high_max - df['Close']) / high_low_range
        
        # CCI (Commodity Channel Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        mad = mad.replace(0, np.nan)
        df['CCI'] = (tp - ma_tp) / (0.015 * mad)
        
        return df
    
    def _compute_volatility_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger Bands and related features."""
        # Bollinger Bands
        period = 20
        df['BB_Middle'] = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        
        for n_std in [1, 2]:
            df[f'BB_Upper_{n_std}'] = df['BB_Middle'] + (rolling_std * n_std)
            df[f'BB_Lower_{n_std}'] = df['BB_Middle'] - (rolling_std * n_std)
            bb_middle_safe = df['BB_Middle'].replace(0, np.nan)
            df[f'BB_Width_{n_std}'] = (df[f'BB_Upper_{n_std}'] - df[f'BB_Lower_{n_std}']) / bb_middle_safe
            bb_range = (df[f'BB_Upper_{n_std}'] - df[f'BB_Lower_{n_std}']).replace(0, np.nan)
            df[f'BB_Position_{n_std}'] = (df['Close'] - df[f'BB_Lower_{n_std}']) / bb_range
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        close_safe = df['Close'].replace(0, np.nan)
        df['ATR_Ratio'] = df['ATR_14'] / close_safe
        
        return df
    
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Relative volume
        df['Relative_Volume'] = df['Volume'] / df['Volume_MA_20']
        
        # Price-volume trend
        df['PVT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume rate of change
        df['Volume_ROC'] = df['Volume'].pct_change(10) * 100
        
        return df
    
    def _compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market regime indicators."""
        # Trend strength (ADX-like)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift()),
            np.abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(window=14).mean().replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
        df['ADX'] = dx.rolling(window=14).mean()
        
        # Volatility regime
        vol_median = df['Volatility_20'].rolling(window=252, min_periods=20).median()
        df['High_Vol_Regime'] = (df['Volatility_20'] > vol_median).astype(int)
        
        # Trend regime
        df['Uptrend'] = (df['Close'] > df['MA_50']).astype(int)
        df['Strong_Uptrend'] = ((df['Close'] > df['MA_50']) & (df['MA_50'] > df['MA_200'])).astype(int)
        
        return df
    
    def _compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute calendar-based features."""
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
        
        # Cyclical encoding for calendar features
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 5)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 5)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get feature names grouped by category.
        
        Returns:
            Dictionary mapping category names to feature lists
        """
        return {
            'price': ['Open', 'High', 'Low', 'Close'],
            'returns': [f for f in self.feature_names if 'Return' in f],
            'volatility': [f for f in self.feature_names if 'Vol' in f or 'ATR' in f],
            'momentum': [f for f in self.feature_names if any(x in f for x in ['RSI', 'MACD', 'Stoch', 'ROC', 'Williams', 'CCI'])],
            'trend': [f for f in self.feature_names if any(x in f for x in ['MA_', 'EMA_', 'ADX', 'trend'])],
            'volume': [f for f in self.feature_names if 'Volume' in f or f in ['PVT', 'OBV']],
            'calendar': [f for f in self.feature_names if any(x in f for x in ['Day', 'Month', 'Quarter', 'Week'])]
        }


class DataNormalizer:
    """
    Data normalization utilities for neural network inputs.
    
    Supports multiple normalization strategies and handles
    proper temporal separation to prevent data leakage.
    """
    
    def __init__(self, method: str = 'standard', feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('standard', 'robust', 'minmax')
            feature_range: Range for minmax scaling
        """
        self.method = method
        self.feature_range = feature_range
        self.scalers: Dict[str, object] = {}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame, feature_columns: List[str]) -> 'DataNormalizer':
        """
        Fit normalizers on training data.
        
        Args:
            data: Training DataFrame
            feature_columns: Columns to normalize
            
        Returns:
            Self for method chaining
        """
        for col in feature_columns:
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler(feature_range=self.feature_range)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            scaler.fit(data[[col]])
            self.scalers[col] = scaler
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted normalizers.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        df = data.copy()
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        
        return df
    
    def fit_transform(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(data, feature_columns)
        return self.transform(data)
    
    def inverse_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform normalized data.
        
        Args:
            data: Normalized DataFrame
            columns: Specific columns to inverse transform (default: all)
            
        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        df = data.copy()
        cols_to_transform = columns or list(self.scalers.keys())
        
        for col in cols_to_transform:
            if col in self.scalers and col in df.columns:
                df[col] = self.scalers[col].inverse_transform(df[[col]])
        
        return df
    
    def get_scaler(self, column: str) -> object:
        """Get scaler for a specific column."""
        return self.scalers.get(column)
    
    def save_state(self) -> Dict:
        """Save normalizer state for reproducibility."""
        state = {
            'method': self.method,
            'feature_range': self.feature_range,
            'is_fitted': self.is_fitted,
            'scaler_params': {}
        }
        
        for col, scaler in self.scalers.items():
            if hasattr(scaler, 'mean_'):
                state['scaler_params'][col] = {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            elif hasattr(scaler, 'center_'):
                state['scaler_params'][col] = {
                    'center': scaler.center_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            elif hasattr(scaler, 'min_'):
                state['scaler_params'][col] = {
                    'min': scaler.min_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
        
        return state
