"""
Utility functions for AMOC ML tuning project.

Helper functions for file I/O, logging, and common operations.
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict, List


def save_model(model, filepath: str):
    """
    Save trained model to file.
    
    Parameters
    ----------
    model : object
        Trained model to save
    filepath : str or Path
        Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Load trained model from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved model
    
    Returns
    -------
    model : object
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {filepath}")
    return model


def save_results(results: Dict, filepath: str):
    """
    Save results dictionary to JSON file.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str or Path
        Path to save the results
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
            results_serializable[key] = float(value)
        else:
            results_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to: {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the results file
    
    Returns
    -------
    results : dict
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def log_experiment(
    experiment_name: str,
    config: Dict,
    metrics: Dict,
    log_file: str = "experiments/experiment_log.csv"
):
    """
    Log experiment details to CSV file.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    config : dict
        Configuration parameters
    metrics : dict
        Performance metrics
    log_file : str or Path
        Path to log file
    """
    import pandas as pd
    
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        **config,
        **metrics
    }
    
    # Append to log file
    df_new = pd.DataFrame([log_entry])
    
    if log_file.exists():
        df_existing = pd.read_csv(log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(log_file, index=False)
    print(f"Experiment logged to: {log_file}")


def create_experiment_name(prefix: str = "exp") -> str:
    """
    Create timestamped experiment name.
    
    Parameters
    ----------
    prefix : str
        Prefix for experiment name
    
    Returns
    -------
    name : str
        Experiment name with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dirs(dirs: List[str]):
    """
    Ensure directories exist, create if they don't.
    
    Parameters
    ----------
    dirs : list of str
        List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def compute_statistics_summary(data: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for array.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    
    Returns
    -------
    stats : dict
        Dictionary of statistics
    """
    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'skewness': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3)),
        'kurtosis': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 4))
    }
    
    return stats


def print_array_info(arr: np.ndarray, name: str = "Array"):
    """
    Print information about numpy array.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to inspect
    name : str
        Name for display
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {np.min(arr):.4f}")
    print(f"  Max: {np.max(arr):.4f}")
    print(f"  Mean: {np.mean(arr):.4f}")
    print(f"  Std: {np.std(arr):.4f}")


def batch_process(data: List, func, batch_size: int = 100, **kwargs):
    """
    Process data in batches with progress tracking.
    
    Parameters
    ----------
    data : list
        List of items to process
    func : callable
        Function to apply to each item
    batch_size : int
        Size of each batch
    **kwargs
        Additional arguments for func
    
    Returns
    -------
    results : list
        List of results
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Install tqdm for progress bars: pip install tqdm")
    
    results = []
    n_batches = (len(data) + batch_size - 1) // batch_size
    
    iterator = range(n_batches)
    if use_tqdm:
        iterator = tqdm(iterator, desc="Processing batches")
    
    for i in iterator:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]
        
        batch_results = [func(item, **kwargs) for item in batch]
        results.extend(batch_results)
    
    return results
