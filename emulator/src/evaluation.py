"""
Model evaluation and cross-validation utilities.

This module provides functions for cross-validation, performance metrics,
and model comparison.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Optional, List, Type


def crossvalidation_bins(
    X: np.ndarray,
    Y: np.ndarray,
    bin_frequencies: np.ndarray,
    emulator_class: Type,
    emulator_params: Optional[Dict] = None,
    cv: int = 5,
    output_names: List[str] = ['mean', 'std', 'q25', 'q75'],
    random_state: int = 42
) -> Dict[str, float]:
    """
    Perform cross-validation for bin-based emulator.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_parameters)
        Input parameters
    Y : np.ndarray, shape (n_samples, n_stats)
        Summary statistics (mean, std, q25, q75)
    bin_frequencies : np.ndarray, shape (n_samples, n_bins)
        Bin frequencies for each sample
    emulator_class : class
        Emulator class (e.g., GPEmulatorBins)
    emulator_params : dict, optional
        Parameters for emulator initialization
    cv : int, default=5
        Number of cross-validation folds
    output_names : list of str
        Names of output statistics
    random_state : int, default=42
        Random state for reproducibility
    
    Returns
    -------
    results : dict
        Cross-validation results containing:
        - bins_rmse: Mean RMSE for bin frequencies
        - bins_max_error: Mean of maximum errors across bins
        - bins_r2: Mean R² for bin frequencies
        - {stat}_rmse: RMSE for each summary statistic
    """
    if emulator_params is None:
        emulator_params = {}
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    bin_rmses = []
    bin_sup_diffs = []  # Maximum error across all bins
    individual_rmses = {name: [] for name in output_names}
    bin_r2_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]  
        bins_train, bins_test = bin_frequencies[train_idx], bin_frequencies[test_idx]
        
        # Fit and predict
        emulator = emulator_class(**emulator_params)
        emulator.fit(X_train, Y_train, bins_train)
        bins_pred, Y_pred = emulator.predict(X_test)
        
        # Bin frequency metrics
        bin_rmse = np.sqrt(mean_squared_error(bins_test.flatten(), bins_pred.flatten()))
        bin_rmses.append(bin_rmse)
        
        # Bin supremum difference (max pointwise error across all test samples)
        bin_pointwise_errors = np.abs(bins_test - bins_pred)
        bin_sup_diff = np.max(bin_pointwise_errors)
        bin_sup_diffs.append(bin_sup_diff)
        
        # Overall R² score
        scores = emulator.score(X_test, Y_test, bins_test)
        bin_r2_scores.append(scores['bin_frequencies_r2'])
        
        # Individual summary statistics RMSEs
        for i, name in enumerate(output_names):
            if i < Y_test.shape[1]:
                rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
                individual_rmses[name].append(rmse)
    
    # Calculate means across all folds
    results = {
        'bins_rmse': np.mean(bin_rmses),
        'bins_max_error': np.mean(bin_sup_diffs),
        'bins_r2': np.mean(bin_r2_scores)
    }
    
    # Add individual statistics means
    for name in output_names:
        if len(individual_rmses[name]) > 0:
            results[f'{name}_rmse'] = np.mean(individual_rmses[name])
    
    return results


def evaluate_model(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    bin_test: np.ndarray,
    stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
) -> Dict[str, float]:
    """
    Evaluate trained model on test set.
    
    Parameters
    ----------
    model : fitted emulator
        Trained model instance
    X_test : np.ndarray, shape (n_samples, n_parameters)
        Test input parameters
    Y_test : np.ndarray, shape (n_samples, n_stats)
        True summary statistics
    bin_test : np.ndarray, shape (n_samples, n_bins)
        True bin frequencies
    stat_names : list of str
        Names of statistics
    
    Returns
    -------
    metrics : dict
        Dictionary of performance metrics including RMSE and R² scores
    """
    # Get predictions
    bin_pred, Y_pred = model.predict(X_test, return_std=False)
    
    # Compute metrics
    metrics = {}
    
    # Bin frequency metrics
    metrics['bin_rmse'] = np.sqrt(mean_squared_error(bin_test, bin_pred))
    metrics['bin_r2'] = r2_score(bin_test.flatten(), bin_pred.flatten())
    
    # Individual statistic metrics
    for i, name in enumerate(stat_names):
        if i < Y_test.shape[1]:
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
            metrics[f'{name}_r2'] = r2_score(Y_test[:, i], Y_pred[:, i])
    
    return metrics


def print_cv_results(results: Dict[str, float], title: str = "Cross-Validation Results"):
    """
    Print cross-validation results in a formatted way.
    
    Parameters
    ----------
    results : dict
        Cross-validation results from crossvalidation_bins
    title : str
        Title to display
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    
    # Bin metrics
    print(f"Bins RMSE: {results['bins_rmse']:.4f}")
    print(f"Bins Max Error (mean of fold maxima): {results['bins_max_error']:.4f}")
    print(f"Bins R²: {results['bins_r2']:.3f}")
    
    # Individual statistics
    print("\nIndividual Statistics RMSE:")
    for key, value in results.items():
        if key.endswith('_rmse') and not key.startswith('bins'):
            stat_name = key.replace('_rmse', '')
            print(f"  {stat_name}: {value:.4f}")
    
    print("=" * 60 + "\n")


def print_test_results(metrics: Dict[str, float], title: str = "Test Set Performance"):
    """
    Print test set evaluation results.
    
    Parameters
    ----------
    metrics : dict
        Test metrics from evaluate_model
    title : str
        Title to display
    """
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    
    # Bin metrics
    print(f"Bin frequencies R²: {metrics['bin_r2']:.3f}")
    print(f"Bin frequencies RMSE: {metrics['bin_rmse']:.4f}")
    
    # Individual statistics
    print("\nIndividual Statistics:")
    stat_names = [key.replace('_rmse', '') for key in metrics.keys() 
                  if key.endswith('_rmse') and not key.startswith('bin')]
    
    for name in stat_names:
        if f'{name}_r2' in metrics and f'{name}_rmse' in metrics:
            print(f"  {name}: R² = {metrics[f'{name}_r2']:.3f}, "
                  f"RMSE = {metrics[f'{name}_rmse']:.4f}")
    
    print("=" * 40 + "\n")
