"""
Model comparison script for GP vs NN emulators.

This script provides easy comparison between Gaussian Process 
and Neural Network emulators on your AMOC data.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

from src.models import GPEmulatorBins, NNEmulatorBins
from src.evaluation import evaluate_model, crossvalidation_bins
from config.config import Config


def compare_models_simple(X, bin_frequencies, other_stats, config=None):
    """
    Simple comparison of GP vs NN on a single train/test split.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    config : Config, optional
        Configuration object
    
    Returns
    -------
    results : dict
        Comparison results for both models
    """
    if config is None:
        config = Config
    
    print("\n" + "="*70)
    print("SIMPLE COMPARISON: GP vs NN (Single Train/Test Split)")
    print("="*70)
    
    # Split data
    X_train, X_test, bin_train, bin_test, Y_train, Y_test = train_test_split(
        X, bin_frequencies, other_stats,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    results = {}
    
    # ========================================
    # Train and evaluate GP
    # ========================================
    print("\n" + "-"*70)
    print("Training Gaussian Process Emulator...")
    print("-"*70)
    
    gp_start = time.time()
    gp_model = GPEmulatorBins(kernel=config.KERNEL)
    gp_model.fit(X_train, Y_train, bin_train)
    gp_train_time = time.time() - gp_start
    
    gp_pred_start = time.time()
    bin_pred_gp, Y_pred_gp, _, _ = gp_model.predict(X_test, return_std=True)
    gp_pred_time = time.time() - gp_pred_start
    
    gp_metrics = evaluate_model(gp_model, X_test, Y_test, bin_test, config.STAT_NAMES)
    
    results['GP'] = {
        'metrics': gp_metrics,
        'train_time': gp_train_time,
        'predict_time': gp_pred_time,
        'predictions': (bin_pred_gp, Y_pred_gp)
    }
    
    print(f"✓ Training time: {gp_train_time:.2f}s")
    print(f"✓ Prediction time: {gp_pred_time:.4f}s")
    print(f"✓ Bins R²: {gp_metrics['bin_r2']:.3f}")
    print(f"✓ Mean R²: {gp_metrics['mean_r2']:.3f}")
    
    # ========================================
    # Train and evaluate NN
    # ========================================
    print("\n" + "-"*70)
    print("Training Neural Network Emulator...")
    print("-"*70)
    
    nn_start = time.time()
    nn_model = NNEmulatorBins(
        n_parameters=X_train.shape[1],
        n_bins=bin_train.shape[1],
        n_other_stats=Y_train.shape[1],
        n_hidden_layers=config.NN_N_HIDDEN_LAYERS,
        n_nodes=config.NN_N_NODES,
        activation=config.NN_ACTIVATION,
        learning_rate=config.NN_LEARNING_RATE,
        l2_reg=config.NN_L2_REG,
        dropout_rate=config.NN_DROPOUT_RATE,
        patience=config.NN_PATIENCE
    )
    nn_model.fit(
        X_train, Y_train, bin_train,
        epochs=config.NN_EPOCHS,
        batch_size=config.NN_BATCH_SIZE,
        validation_split=config.NN_VALIDATION_SPLIT,
        verbose=0
    )
    nn_train_time = time.time() - nn_start
    
    nn_pred_start = time.time()
    bin_pred_nn, Y_pred_nn = nn_model.predict(X_test)
    nn_pred_time = time.time() - nn_pred_start
    
    nn_metrics = evaluate_model(nn_model, X_test, Y_test, bin_test, config.STAT_NAMES)
    
    results['NN'] = {
        'metrics': nn_metrics,
        'train_time': nn_train_time,
        'predict_time': nn_pred_time,
        'predictions': (bin_pred_nn, Y_pred_nn)
    }
    
    print(f"✓ Training time: {nn_train_time:.2f}s")
    print(f"✓ Prediction time: {nn_pred_time:.4f}s")
    print(f"✓ Bins R²: {nn_metrics['bin_r2']:.3f}")
    print(f"✓ Mean R²: {nn_metrics['mean_r2']:.3f}")
    
    # ========================================
    # Summary comparison
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'GP':<20} {'NN':<20} {'Winner':<10}")
    print("-"*70)
    
    # Training time
    faster_train = 'NN' if nn_train_time < gp_train_time else 'GP'
    print(f"{'Training Time (s)':<25} {gp_train_time:<20.2f} {nn_train_time:<20.2f} {faster_train:<10}")
    
    # Prediction time
    faster_pred = 'NN' if nn_pred_time < gp_pred_time else 'GP'
    speedup = gp_pred_time / nn_pred_time if nn_pred_time > 0 else 0
    print(f"{'Prediction Time (s)':<25} {gp_pred_time:<20.4f} {nn_pred_time:<20.4f} {faster_pred:<10} ({speedup:.0f}x)")
    
    # Bins R²
    better_bins = 'GP' if gp_metrics['bin_r2'] > nn_metrics['bin_r2'] else 'NN'
    print(f"{'Bins R²':<25} {gp_metrics['bin_r2']:<20.3f} {nn_metrics['bin_r2']:<20.3f} {better_bins:<10}")
    
    # Mean R²
    better_mean = 'GP' if gp_metrics['mean_r2'] > nn_metrics['mean_r2'] else 'NN'
    print(f"{'Mean R²':<25} {gp_metrics['mean_r2']:<20.3f} {nn_metrics['mean_r2']:<20.3f} {better_mean:<10}")
    
    # Bins RMSE
    better_rmse = 'GP' if gp_metrics['bin_rmse'] < nn_metrics['bin_rmse'] else 'NN'
    print(f"{'Bins RMSE':<25} {gp_metrics['bin_rmse']:<20.4f} {nn_metrics['bin_rmse']:<20.4f} {better_rmse:<10}")
    
    print(f"\n{'Uncertainty Available':<25} {'✓ Yes':<20} {'✗ No':<20}")
    
    return results


def compare_models_cv(X, bin_frequencies, other_stats, config=None):
    """
    Comprehensive comparison using cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    config : Config, optional
        Configuration object
    
    Returns
    -------
    results : dict
        Cross-validation results for both models
    """
    if config is None:
        config = Config
    
    print("\n" + "="*70)
    print(f"COMPREHENSIVE COMPARISON: {config.CV_FOLDS}-Fold Cross-Validation")
    print("="*70)
    
    results = {}
    
    # GP Cross-validation
    print("\n" + "-"*70)
    print("GP Cross-Validation...")
    print("-"*70)
    
    gp_params = {'kernel': config.KERNEL}
    gp_cv_results = crossvalidation_bins(
        X, other_stats, bin_frequencies,
        GPEmulatorBins,
        emulator_params=gp_params,
        cv=config.CV_FOLDS,
        output_names=config.STAT_NAMES,
        random_state=config.RANDOM_STATE
    )
    results['GP'] = gp_cv_results
    
    print(f"✓ Bins RMSE: {gp_cv_results['bins_rmse']:.4f}")
    print(f"✓ Bins R²: {gp_cv_results['bins_r2']:.3f}")
    print(f"✓ Mean RMSE: {gp_cv_results['mean_rmse']:.4f}")
    
    # NN Cross-validation
    print("\n" + "-"*70)
    print("NN Cross-Validation...")
    print("-"*70)
    
    nn_params = {
        'n_parameters': X.shape[1],
        'n_bins': bin_frequencies.shape[1],
        'n_other_stats': other_stats.shape[1],
        'n_hidden_layers': config.NN_N_HIDDEN_LAYERS,
        'n_nodes': config.NN_N_NODES,
        'activation': config.NN_ACTIVATION,
        'learning_rate': config.NN_LEARNING_RATE,
        'l2_reg': config.NN_L2_REG,
        'dropout_rate': config.NN_DROPOUT_RATE,
        'patience': config.NN_PATIENCE
    }
    nn_cv_results = crossvalidation_bins(
        X, other_stats, bin_frequencies,
        NNEmulatorBins,
        emulator_params=nn_params,
        cv=config.CV_FOLDS,
        output_names=config.STAT_NAMES,
        random_state=config.RANDOM_STATE
    )
    results['NN'] = nn_cv_results
    
    print(f"✓ Bins RMSE: {nn_cv_results['bins_rmse']:.4f}")
    print(f"✓ Bins R²: {nn_cv_results['bins_r2']:.3f}")
    print(f"✓ Mean RMSE: {nn_cv_results['mean_rmse']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'GP':<20} {'NN':<20} {'Winner':<10}")
    print("-"*70)
    
    better_rmse = 'GP' if gp_cv_results['bins_rmse'] < nn_cv_results['bins_rmse'] else 'NN'
    print(f"{'Bins RMSE':<25} {gp_cv_results['bins_rmse']:<20.4f} {nn_cv_results['bins_rmse']:<20.4f} {better_rmse:<10}")
    
    better_r2 = 'GP' if gp_cv_results['bins_r2'] > nn_cv_results['bins_r2'] else 'NN'
    print(f"{'Bins R²':<25} {gp_cv_results['bins_r2']:<20.3f} {nn_cv_results['bins_r2']:<20.3f} {better_r2:<10}")
    
    better_mean = 'GP' if gp_cv_results['mean_rmse'] < nn_cv_results['mean_rmse'] else 'NN'
    print(f"{'Mean RMSE':<25} {gp_cv_results['mean_rmse']:<20.4f} {nn_cv_results['mean_rmse']:<20.4f} {better_mean:<10}")
    
    return results


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare GP vs NN emulators')
    parser.add_argument(
        '--mode',
        type=str,
        default='simple',
        choices=['simple', 'cv', 'both'],
        help='Comparison mode: simple (train/test split), cv (cross-validation), or both'
    )
    
    args = parser.parse_args()
    
    Config.display()
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print("\nWARNING: Please load your actual data.")
    print("Expected format:")
    print("  - X: (n_samples, n_parameters)")
    print("  - bin_frequencies: (n_samples, n_bins)")
    print("  - other_stats: (n_samples, 4)")
    
    # TODO: Load your data
    # X = np.load('data/processed/input_parameters.npy')
    # bin_frequencies = np.load('data/processed/bin_frequencies.npy')
    # other_stats = np.load('data/processed/other_stats.npy')
    
    # if args.mode in ['simple', 'both']:
    #     results_simple = compare_models_simple(X, bin_frequencies, other_stats, Config)
    
    # if args.mode in ['cv', 'both']:
    #     results_cv = compare_models_cv(X, bin_frequencies, other_stats, Config)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
