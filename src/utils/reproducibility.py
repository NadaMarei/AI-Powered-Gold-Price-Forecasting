"""
Reproducibility Utilities
=========================

Ensure experiment reproducibility through:
- Seed management
- Configuration hashing
- Environment logging
"""

import torch
import numpy as np
import random
import hashlib
import json
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Enable deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                logger.warning(f"Could not enable deterministic algorithms: {e}")
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set all random seeds to {seed}")


def get_experiment_hash(config: Dict) -> str:
    """
    Generate a unique hash for experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Hash string
    """
    # Convert config to stable string representation
    config_str = json.dumps(config, sort_keys=True, default=str)
    
    # Generate SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode())
    return hash_obj.hexdigest()[:12]


def save_experiment_config(
    config: Dict,
    output_dir: str = "experiments",
    experiment_name: Optional[str] = None
) -> str:
    """
    Save experiment configuration for reproducibility.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory
        experiment_name: Optional experiment name
        
    Returns:
        Path to saved configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment ID
    exp_hash = get_experiment_hash(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_id = f"{experiment_name}_{timestamp}_{exp_hash}"
    else:
        exp_id = f"exp_{timestamp}_{exp_hash}"
    
    # Create experiment directory
    exp_dir = output_path / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Save environment info
    env_info = get_environment_info()
    env_path = exp_dir / "environment.json"
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    logger.info(f"Saved experiment config to {exp_dir}")
    
    return str(exp_dir)


def get_environment_info() -> Dict[str, Any]:
    """
    Get environment information for reproducibility.
    
    Returns:
        Dictionary with environment details
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
    
    # Package versions
    try:
        import pandas
        info['pandas_version'] = pandas.__version__
    except ImportError:
        pass
    
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import shap
        info['shap_version'] = shap.__version__
    except ImportError:
        pass
    
    return info


def load_experiment_config(exp_dir: str) -> Dict:
    """
    Load experiment configuration from directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(exp_dir) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


class ExperimentTracker:
    """
    Track experiment runs for reproducibility.
    """
    
    def __init__(self, base_dir: str = "experiments"):
        """
        Initialize tracker.
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_exp_dir: Optional[Path] = None
        self.run_history: List[Dict] = []
    
    def start_experiment(
        self,
        config: Dict,
        name: Optional[str] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            config: Experiment configuration
            name: Optional experiment name
            
        Returns:
            Experiment directory path
        """
        exp_dir = save_experiment_config(config, str(self.base_dir), name)
        self.current_exp_dir = Path(exp_dir)
        
        # Log start
        self.run_history.append({
            'experiment': exp_dir,
            'start_time': datetime.now().isoformat(),
            'config': config
        })
        
        return exp_dir
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for current experiment.
        
        Args:
            metrics: Metric values
            step: Optional step number
        """
        if self.current_exp_dir is None:
            logger.warning("No active experiment")
            return
        
        metrics_file = self.current_exp_dir / "metrics.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def save_artifact(self, name: str, data: Any):
        """
        Save an artifact (model, figure, etc.).
        
        Args:
            name: Artifact name
            data: Data to save
        """
        if self.current_exp_dir is None:
            logger.warning("No active experiment")
            return
        
        artifacts_dir = self.current_exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        artifact_path = artifacts_dir / name
        
        if isinstance(data, torch.nn.Module):
            torch.save(data.state_dict(), f"{artifact_path}.pt")
        elif isinstance(data, dict):
            with open(f"{artifact_path}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, np.ndarray):
            np.save(f"{artifact_path}.npy", data)
        else:
            logger.warning(f"Unknown data type for artifact: {type(data)}")
    
    def end_experiment(self, status: str = "completed"):
        """
        End current experiment.
        
        Args:
            status: Experiment status
        """
        if self.current_exp_dir is None:
            return
        
        # Save final status
        status_file = self.current_exp_dir / "status.json"
        with open(status_file, 'w') as f:
            json.dump({
                'status': status,
                'end_time': datetime.now().isoformat()
            }, f)
        
        # Update run history
        if self.run_history:
            self.run_history[-1]['end_time'] = datetime.now().isoformat()
            self.run_history[-1]['status'] = status
        
        self.current_exp_dir = None
