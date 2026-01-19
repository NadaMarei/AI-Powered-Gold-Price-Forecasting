"""
PyTorch Dataset Module for Time Series
======================================

Implements PyTorch datasets and data loaders for sequential gold price data
with proper temporal handling to prevent data leakage.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GoldPriceDataset(Dataset):
    """
    PyTorch Dataset for gold price time series.
    
    Creates sequences of features and corresponding targets for
    training sequential models like GRU/LSTM.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'Close',
        sequence_length: int = 60,
        forecast_horizon: int = 1,
        return_dates: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data: DataFrame with all features
            feature_columns: List of column names to use as features
            target_column: Column name for target variable
            sequence_length: Number of time steps in each sequence
            forecast_horizon: Number of steps ahead to predict
            return_dates: Whether to return date indices
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.return_dates = return_dates
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Extract features and target
        self.features = data[feature_columns].values.astype(np.float32)
        self.targets = data[target_column].values.astype(np.float32)
        self.dates = data.index
        
        # Compute valid indices
        self.valid_indices = self._compute_valid_indices()
        
        logger.info(f"Created dataset with {len(self.valid_indices)} valid sequences")
        
    def _compute_valid_indices(self) -> List[int]:
        """
        Compute indices where we can create valid sequences.
        
        Returns:
            List of valid starting indices
        """
        total_length = len(self.features)
        required_length = self.sequence_length + self.forecast_horizon
        
        if total_length < required_length:
            raise ValueError(
                f"Data length ({total_length}) is shorter than required "
                f"sequence length + forecast horizon ({required_length})"
            )
        
        # Valid indices are those where we can create a full sequence + target
        valid_indices = list(range(total_length - required_length + 1))
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence-target pair.
        
        Args:
            idx: Index in valid_indices
            
        Returns:
            Tuple of (sequence, target) tensors
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.forecast_horizon - 1
        
        # Get sequence features: [sequence_length, num_features]
        sequence = torch.from_numpy(self.features[start_idx:end_idx])
        
        # Get target value
        target = torch.tensor(self.targets[target_idx], dtype=torch.float32)
        
        if self.return_dates:
            date = self.dates[target_idx]
            return sequence, target, date
        
        return sequence, target
    
    def get_full_sequence(self, idx: int) -> Dict:
        """
        Get full sequence information for analysis.
        
        Args:
            idx: Index in valid_indices
            
        Returns:
            Dictionary with sequence, target, and metadata
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        target_idx = end_idx + self.forecast_horizon - 1
        
        return {
            'sequence': self.features[start_idx:end_idx],
            'target': self.targets[target_idx],
            'sequence_dates': self.dates[start_idx:end_idx],
            'target_date': self.dates[target_idx],
            'feature_names': self.feature_columns
        }


class SequenceCollator:
    """
    Custom collator for batching sequences with variable requirements.
    """
    
    def __init__(self, include_volatility: bool = True, volatility_idx: Optional[int] = None):
        """
        Initialize collator.
        
        Args:
            include_volatility: Whether to include volatility for regime-aware models
            volatility_idx: Index of volatility feature in feature array
        """
        self.include_volatility = include_volatility
        self.volatility_idx = volatility_idx
    
    def __call__(self, batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of sequences.
        
        Args:
            batch: List of (sequence, target) tuples
            
        Returns:
            Dictionary with batched tensors
        """
        sequences = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        
        result = {
            'sequences': sequences,
            'targets': targets
        }
        
        if self.include_volatility and self.volatility_idx is not None:
            # Extract volatility from last time step
            volatility = sequences[:, -1, self.volatility_idx]
            result['volatility'] = volatility
        
        return result


def create_data_loaders(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'Close',
    sequence_length: int = 60,
    forecast_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 0,
    volatility_column: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.
    
    Uses temporal split to prevent data leakage.
    
    Args:
        data: Full DataFrame with features
        feature_columns: Columns to use as features
        target_column: Column to predict
        sequence_length: Length of input sequences
        forecast_horizon: Steps ahead to predict
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        volatility_column: Column containing volatility for regime detection
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, split_info)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    total_samples = len(data)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    # Temporal split
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Get volatility index for regime-aware collator
    volatility_idx = None
    if volatility_column and volatility_column in feature_columns:
        volatility_idx = feature_columns.index(volatility_column)
    
    # Create datasets
    train_dataset = GoldPriceDataset(
        train_data, feature_columns, target_column,
        sequence_length, forecast_horizon
    )
    val_dataset = GoldPriceDataset(
        val_data, feature_columns, target_column,
        sequence_length, forecast_horizon
    )
    test_dataset = GoldPriceDataset(
        test_data, feature_columns, target_column,
        sequence_length, forecast_horizon
    )
    
    # Create collator
    collator = SequenceCollator(
        include_volatility=volatility_idx is not None,
        volatility_idx=volatility_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    split_info = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'train_date_range': (train_data.index.min(), train_data.index.max()),
        'val_date_range': (val_data.index.min(), val_data.index.max()),
        'test_date_range': (test_data.index.min(), test_data.index.max()),
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'num_features': len(feature_columns)
    }
    
    return train_loader, val_loader, test_loader, split_info
