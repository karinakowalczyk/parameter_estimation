"""
Data preparation module for AMOC time series analysis.

This module contains functions for computing summary statistics,
bin frequencies, and data preprocessing for ML emulators.
"""

import numpy as np
import xarray as xr
from typing import Dict, Optional, Tuple, List


def get_amoc_bins(n_bins: int = 10) -> np.ndarray:
    """
    Return the standard AMOC bins used across all functions.
    
    Parameters
    ----------
    n_bins : int, default=10
        Number of bins to create
    
    Returns
    -------
    np.ndarray
        Bin edges from 0 to 35 Sv
    """
    return np.linspace(0, 35, n_bins + 1)


def compute_simple_summary_stats(
    amoc_data: np.ndarray,
    time_data: Optional[np.ndarray] = None,
    remove_spinup: bool = False,
    spinup_fraction: float = 0.1,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute simple summary statistics for AMOC time series.
    
    Parameters
    ----------
    amoc_data : array-like
        AMOC strength time series
    time_data : array-like, optional
        Time values (if None, assumes uniform spacing)
    remove_spinup : bool, default=False
        Whether to remove initial spinup period
    spinup_fraction : float, default=0.1
        Fraction of data to remove as spinup (default 0.1 = 10%)
    n_bins : int, default=10
        Number of bins for histogram
    
    Returns
    -------
    dict
        Dictionary containing:
        - mean: Mean AMOC strength
        - std: Standard deviation
        - q25: 25th percentile
        - q75: 75th percentile
        - bin_frequencies: Normalized histogram frequencies
    """
    amoc_data = np.asarray(amoc_data)
    
    # Remove spinup if requested
    if remove_spinup:
        start_idx = int(len(amoc_data) * spinup_fraction)
        amoc_data = amoc_data[start_idx:]
    
    # Define fixed bins for consistency across all simulations
    bins = get_amoc_bins(n_bins)
    
    # Calculate bin frequencies for ML emulator
    bin_counts, _ = np.histogram(amoc_data, bins=bins)
    bin_frequencies = bin_counts / len(amoc_data)  # Normalize to [0,1]
    
    # Compute summary statistics
    stats = {
        'mean': np.mean(amoc_data),
        'std': np.std(amoc_data),
        'q25': np.percentile(amoc_data, 25),
        'q75': np.percentile(amoc_data, 75),
        'bin_frequencies': bin_frequencies,
    }
    
    return stats


def load_amoc_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AMOC data from netCDF file.
    
    Parameters
    ----------
    filepath : str
        Path to netCDF file containing AMOC data
    
    Returns
    -------
    amoc : np.ndarray
        AMOC strength time series
    time : np.ndarray
        Time values
    """
    ds = xr.open_dataset(filepath)
    amoc = ds.amoc26N.values
    time = ds.time.values
    ds.close()
    
    return amoc, time


def process_ensemble_data(
    model_files: List[str],
    remove_spinup: bool = False,
    n_bins: int = 10
) -> List[Dict[str, np.ndarray]]:
    """
    Process multiple ensemble runs and compute statistics for each.
    
    Parameters
    ----------
    model_files : list of str
        List of file paths to ensemble run netCDF files
    remove_spinup : bool, default=False
        Whether to remove spinup period
    n_bins : int, default=10
        Number of bins for histograms
    
    Returns
    -------
    list of dict
        List of statistics dictionaries, one per ensemble run
    """
    ensemble_stats = []
    
    for file in model_files:
        amoc, time = load_amoc_data(file)
        stats = compute_simple_summary_stats(
            amoc, time, 
            remove_spinup=remove_spinup,
            n_bins=n_bins
        )
        ensemble_stats.append(stats)
    
    return ensemble_stats


def prepare_ml_data(
    ensemble_stats: List[Dict[str, np.ndarray]],
    stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for ML model training.
    
    Parameters
    ----------
    ensemble_stats : list of dict
        List of statistics from compute_simple_summary_stats
    stat_names : list of str
        Names of statistics to extract
    
    Returns
    -------
    bin_frequencies : np.ndarray, shape (n_samples, n_bins)
        Bin frequency matrix
    other_stats : np.ndarray, shape (n_samples, n_stats)
        Other summary statistics matrix
    """
    bin_frequencies = np.array([stats['bin_frequencies'] for stats in ensemble_stats])
    other_stats = np.array([[stats[name] for name in stat_names] for stats in ensemble_stats])
    
    return bin_frequencies, other_stats
