"""
Visualization utilities for AMOC analysis and model predictions.

This module provides plotting functions for time series, histograms,
and model prediction comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import List, Dict, Optional
from src.data_preparation import get_amoc_bins


def plot_amoc_ensemble_simple(
    ensemble_stats: List[Dict],
    model_files: List[str],
    amoc_default: np.ndarray,
    default_stats: Dict,
    n_runs: Optional[int] = None,
    figsize_per_run: float = 3,
    n_bins: int = 10
):
    """
    Plot AMOC time series and histograms for ensemble runs.
    
    Parameters
    ----------
    ensemble_stats : list of dict
        List of stats dictionaries from compute_simple_summary_stats
    model_files : list of str
        List of file paths for ensemble runs
    amoc_default : np.ndarray
        Default/reference AMOC time series
    default_stats : dict
        Stats dictionary for default run
    n_runs : int, optional
        Number of runs to plot (if None, plots all)
    figsize_per_run : float, default=3
        Figure height per run in inches
    n_bins : int, default=10
        Number of histogram bins
    """
    if n_runs is None:
        n_runs = len(ensemble_stats)
    
    fig, axes = plt.subplots(
        n_runs, 2, 
        figsize=(12, figsize_per_run * n_runs),
        gridspec_kw={'width_ratios': [3, 2]}
    )
    
    # Handle single run case
    if n_runs == 1:
        axes = [axes]
    
    bins = get_amoc_bins(n_bins)
    
    for i, (stats, file) in enumerate(zip(ensemble_stats[:n_runs], model_files[:n_runs])):
        ds = xr.open_dataset(file)
        amoc = ds.amoc26N.values
        ds.close()
        
        # ----------------------------
        # 1. Time series plot
        # ----------------------------
        axes[i][0].plot(amoc, color='darkblue', alpha=0.8, label='Ensemble')
        axes[i][0].plot(amoc_default, color='lightcoral', alpha=0.8, label='Default')
        
        # Add horizontal lines for key statistics
        axes[i][0].axhline(stats['mean'], color='darkblue', linestyle='-', 
                          alpha=0.3, label='Mean (Ens)')
        axes[i][0].axhline(default_stats['mean'], color='lightcoral', 
                          linestyle='-', alpha=0.3, label='Mean (Def)')
        axes[i][0].axhline(stats['q25'], color='darkblue', linestyle='--', alpha=0.3)
        axes[i][0].axhline(stats['q75'], color='darkblue', linestyle='--', alpha=0.3)
        
        axes[i][0].set_ylabel('AMOC (Sv)')
        axes[i][0].set_xlabel('Time')
        axes[i][0].set_title(f'AMOC Time Series - Run {i+1}')
        axes[i][0].legend()
        axes[i][0].grid(alpha=0.3)
        
        # Add summary stats text box
        stats_text = (
            f"Mean: {stats['mean']:.1f} Sv (Def: {default_stats['mean']:.1f})\n"
            f"Std: {stats['std']:.1f} Sv (Def: {default_stats['std']:.1f})\n"
            f"Q25-Q75: {stats['q25']:.1f}-{stats['q75']:.1f} Sv\n"
        )
        
        axes[i][0].text(
            0.98, 0.02, stats_text,
            transform=axes[i][0].transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        )
        
        # ----------------------------
        # 2. Histogram plot
        # ----------------------------
        axes[i][1].hist(amoc, bins=bins, alpha=0.7, color='darkblue', 
                       label='Ensemble', density=True, edgecolor='black')
        axes[i][1].hist(amoc_default, bins=bins, alpha=0.5, color='lightcoral', 
                       label='Default', density=True, edgecolor='black')
        
        # Add vertical lines for key statistics
        axes[i][1].axvline(stats['mean'], color='darkblue', linestyle='-', 
                          alpha=0.8, label='Mean (Ens)')
        axes[i][1].axvline(default_stats['mean'], color='lightcoral', 
                          linestyle='-', alpha=0.8, label='Mean (Def)')
        axes[i][1].axvline(stats['q25'], color='darkblue', linestyle='--', 
                          alpha=0.7, label='Q25/Q75 (Ens)')
        axes[i][1].axvline(stats['q75'], color='darkblue', linestyle='--', alpha=0.7)
        
        axes[i][1].set_xlabel('AMOC (Sv)')
        axes[i][1].set_ylabel('Density')
        axes[i][1].set_title(f'AMOC Distribution - Run {i+1}')
        axes[i][1].legend()
        axes[i][1].grid(alpha=0.3)
        axes[i][1].set_xlim(0, 35)
    
    plt.tight_layout()
    plt.show()


def plot_prediction_comparison(
    bin_test: np.ndarray,
    bin_pred: np.ndarray,
    bin_std: np.ndarray,
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    n_bins: int = 10,
    n_samples: int = 3,
    stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
):
    """
    Plot comparison of true vs predicted bin frequencies.
    
    Parameters
    ----------
    bin_test : np.ndarray, shape (n_test_samples, n_bins)
        True bin frequencies
    bin_pred : np.ndarray, shape (n_test_samples, n_bins)
        Predicted bin frequencies
    bin_std : np.ndarray, shape (n_test_samples, n_bins)
        Prediction uncertainties
    Y_test : np.ndarray, shape (n_test_samples, n_stats)
        True summary statistics
    Y_pred : np.ndarray, shape (n_test_samples, n_stats)
        Predicted summary statistics
    n_bins : int, default=10
        Number of bins
    n_samples : int, default=3
        Number of test samples to plot
    stat_names : list of str
        Names of summary statistics
    """
    bins = get_amoc_bins(n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    n_samples = min(n_samples, len(bin_test))
    
    for i in range(n_samples):
        plt.figure(figsize=(8, 4))
        
        # Plot overlapping bars at same x-coordinates
        plt.bar(bin_centers, bin_test[i], width=bin_width*0.8, 
                alpha=0.6, color='blue', label='True', edgecolor='black')
        plt.bar(bin_centers, bin_pred[i], width=bin_width*0.8, 
                alpha=0.6, color='red', label='Predicted', edgecolor='black')
        
        # Add line plots connecting the top centers of each bar type
        plt.plot(bin_centers, bin_test[i], 'o-', color='blue', 
                 linewidth=2, markersize=6, alpha=0.8)
        plt.plot(bin_centers, bin_pred[i], 'o-', color='red', 
                 linewidth=2, markersize=6, alpha=0.8)
        
        # Add error bars for uncertainty on predicted values
        plt.errorbar(bin_centers, bin_pred[i], yerr=bin_std[i], 
                    fmt='none', color='darkred', capsize=3, alpha=0.7, linewidth=1.5)
        
        plt.xlabel('AMOC (Sv)')
        plt.ylabel('Frequency')
        plt.title(f'Test Sample {i+1} Distribution Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add statistics text
        stats_text = (f"True: μ={Y_test[i,0]:.1f}, σ={Y_test[i,1]:.1f}\n"
                     f"Pred: μ={Y_pred[i,0]:.1f}, σ={Y_pred[i,1]:.1f}")
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    metric_name: str = "R² Score"
):
    """
    Plot learning curve for model training.
    
    Parameters
    ----------
    train_sizes : np.ndarray
        Number of training samples
    train_scores : np.ndarray
        Training scores
    test_scores : np.ndarray
        Test scores
    metric_name : str
        Name of the metric being plotted
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training', linewidth=2)
    plt.plot(train_sizes, test_scores, 'o-', label='Test', linewidth=2)
    plt.xlabel('Training Set Size')
    plt.ylabel(metric_name)
    plt.title(f'Learning Curve: {metric_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prediction_comparison_no_std(
    bin_test: np.ndarray,
    bin_pred: np.ndarray,
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    n_bins: int = 10,
    n_samples: int = 3,
    stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
):
    """
    Plot comparison of true vs predicted bin frequencies (without uncertainties).
    
    This version is for models like Neural Networks that don't provide
    prediction uncertainties.
    
    Parameters
    ----------
    bin_test : np.ndarray, shape (n_test_samples, n_bins)
        True bin frequencies
    bin_pred : np.ndarray, shape (n_test_samples, n_bins)
        Predicted bin frequencies
    Y_test : np.ndarray, shape (n_test_samples, n_stats)
        True summary statistics
    Y_pred : np.ndarray, shape (n_test_samples, n_stats)
        Predicted summary statistics
    n_bins : int, default=10
        Number of bins
    n_samples : int, default=3
        Number of test samples to plot
    stat_names : list of str
        Names of summary statistics
    """
    bins = get_amoc_bins(n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    n_samples = min(n_samples, len(bin_test))
    
    for i in range(n_samples):
        plt.figure(figsize=(8, 4))
        
        # Plot overlapping bars at same x-coordinates
        plt.bar(bin_centers, bin_test[i], width=bin_width*0.8, 
                alpha=0.6, color='blue', label='True', edgecolor='black')
        plt.bar(bin_centers, bin_pred[i], width=bin_width*0.8, 
                alpha=0.6, color='red', label='Predicted', edgecolor='black')
        
        # Add line plots connecting the top centers of each bar type
        plt.plot(bin_centers, bin_test[i], 'o-', color='blue', 
                 linewidth=2, markersize=6, alpha=0.8)
        plt.plot(bin_centers, bin_pred[i], 'o-', color='red', 
                 linewidth=2, markersize=6, alpha=0.8)
        
        plt.xlabel('AMOC (Sv)')
        plt.ylabel('Frequency')
        plt.title(f'Test Sample {i+1} Distribution Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Add statistics text (no uncertainties for NN)
        stats_text = (f"True: μ={Y_test[i,0]:.1f}, σ={Y_test[i,1]:.1f}\n"
                     f"Pred: μ={Y_pred[i,0]:.1f}, σ={Y_pred[i,1]:.1f}")
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()