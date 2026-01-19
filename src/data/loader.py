"""
Data Loader Module for Gold Price Forecasting
==============================================

Handles data acquisition from various sources including Yahoo Finance
and ensures data quality for research-grade analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class GoldPriceDataLoader:
    """
    Robust data loader for gold price and related financial data.
    
    Supports multiple data sources and includes comprehensive validation
    to ensure data quality for research applications.
    """
    
    def __init__(
        self,
        ticker: str = "GC=F",
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            ticker: Yahoo Finance ticker symbol for gold
            start_date: Start date for data retrieval
            end_date: End date (defaults to today)
            cache_dir: Directory for caching downloaded data
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.cache_dir = cache_dir
        self._raw_data: Optional[pd.DataFrame] = None
        
    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch gold price data from Yahoo Finance.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        
        try:
            # Download data from Yahoo Finance
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data retrieved for ticker {self.ticker}")
            
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Ensure datetime index
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            # Basic data validation
            self._validate_data(data)
            
            self._raw_data = data
            logger.info(f"Successfully loaded {len(data)} records")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {str(e)}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate data quality.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data fails validation
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = set(required_columns) - set(data.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for excessive missing values
        missing_pct = data[required_columns].isnull().mean()
        if (missing_pct > 0.1).any():
            logger.warning(f"High missing value percentage detected: {missing_pct.to_dict()}")
        
        # Check for data anomalies
        if (data['High'] < data['Low']).any():
            logger.warning("Data anomaly detected: High < Low on some days")
        
        if (data['Close'] <= 0).any():
            raise ValueError("Invalid negative or zero prices detected")
    
    def get_additional_features(self) -> pd.DataFrame:
        """
        Fetch additional macroeconomic and market features.
        
        Returns:
            DataFrame with additional features aligned to gold price dates
        """
        if self._raw_data is None:
            self.fetch_data()
        
        additional_data = {}
        
        # US Dollar Index (correlated with gold)
        try:
            dxy = yf.download("DX-Y.NYB", start=self.start_date, end=self.end_date, progress=False)
            if not dxy.empty:
                if isinstance(dxy.columns, pd.MultiIndex):
                    dxy.columns = dxy.columns.get_level_values(0)
                additional_data['DXY'] = dxy['Close']
        except Exception as e:
            logger.warning(f"Failed to fetch DXY data: {e}")
        
        # S&P 500 (risk sentiment)
        try:
            spy = yf.download("SPY", start=self.start_date, end=self.end_date, progress=False)
            if not spy.empty:
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                additional_data['SPY'] = spy['Close']
        except Exception as e:
            logger.warning(f"Failed to fetch SPY data: {e}")
        
        # VIX (volatility/fear index)
        try:
            vix = yf.download("^VIX", start=self.start_date, end=self.end_date, progress=False)
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                additional_data['VIX'] = vix['Close']
        except Exception as e:
            logger.warning(f"Failed to fetch VIX data: {e}")
        
        # Treasury Yields (10-year)
        try:
            tnx = yf.download("^TNX", start=self.start_date, end=self.end_date, progress=False)
            if not tnx.empty:
                if isinstance(tnx.columns, pd.MultiIndex):
                    tnx.columns = tnx.columns.get_level_values(0)
                additional_data['TNX'] = tnx['Close']
        except Exception as e:
            logger.warning(f"Failed to fetch TNX data: {e}")
        
        if additional_data:
            additional_df = pd.DataFrame(additional_data)
            # Align to gold price dates
            additional_df = additional_df.reindex(self._raw_data.index)
            return additional_df
        
        return pd.DataFrame(index=self._raw_data.index)
    
    def get_complete_dataset(self) -> pd.DataFrame:
        """
        Get complete dataset with gold prices and additional features.
        
        Returns:
            DataFrame with all features
        """
        if self._raw_data is None:
            self.fetch_data()
        
        # Start with raw gold data
        complete_data = self._raw_data.copy()
        
        # Add additional features
        additional = self.get_additional_features()
        if not additional.empty:
            complete_data = pd.concat([complete_data, additional], axis=1)
        
        # Handle missing values
        complete_data = complete_data.ffill().bfill()
        
        return complete_data
    
    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        """Get the raw data if available."""
        return self._raw_data
    
    def get_data_statistics(self) -> Dict:
        """
        Calculate descriptive statistics for the dataset.
        
        Returns:
            Dictionary with data statistics
        """
        if self._raw_data is None:
            self.fetch_data()
        
        stats = {
            'start_date': self._raw_data.index.min().strftime("%Y-%m-%d"),
            'end_date': self._raw_data.index.max().strftime("%Y-%m-%d"),
            'total_records': len(self._raw_data),
            'trading_days': len(self._raw_data),
            'price_stats': self._raw_data['Close'].describe().to_dict(),
            'volume_stats': self._raw_data['Volume'].describe().to_dict(),
            'missing_values': self._raw_data.isnull().sum().to_dict()
        }
        
        return stats
