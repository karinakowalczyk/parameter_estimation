"""
Hyperparameter tuning script for AMOC emulators.

This script performs systematic hyperparameter search for both
GP and NN models using GridSearch or RandomizedSearch.
"""

import numpy as np
import argparse
from pathlib import Path

from src.tuning import (
    GridSearchCV,
    RandomizedSearchCV,
    tune_gp_kernel,
    tune_nn_architecture
)
from src.models import GPEmulatorBins, NNEmulatorBins
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from config.config import Config


def tune_gp_custom(
    X, bin_frequencies, other_stats,
    search_type='grid',
    n_iter=10
):
    """
    Custom GP hyperparameter tuning with more control.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    search_type : str
        'grid' or 'random'
    n_iter : int
        Number of random iterations
    
    Returns
    -------
    tuner : fitted tuner
    """
    print("\n" + "="*70)
    print("GP HYPERPARAMETER TUNING")
    print("="*70)
    
    # Define search space
    param_grid = {
        'kernel': [
            # Matern kernels with different smoothness
            Matern(length_scale=0.5, nu=0.5),
            Matern(length_scale=0.5, nu=1.5),
            Matern(length_scale=0.5, nu=2.5),
            Matern(length_scale=1.0, nu=0.5),
            Matern(length_scale=1.0, nu=1.5),
            Matern(length_scale=1.0, nu=2.5),
            Matern(length_scale=2.0, nu=0.5),
            Matern(length_scale=2.0, nu=1.5),
            Matern(length_scale=2.0, nu=2.5),
            # RBF kernels
            RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-6),
            RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6),
            RBF(length_scale=2.0) + WhiteKernel(noise_level=1e-6),
            RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
            RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4),
        ]
    }
    
    if search_type == 'grid':
        tuner = GridSearchCV(
            GPEmulatorBins,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring='bins_r2',
            random_state=Config.RANDOM_STATE,
            verbose=1
        )
    else:
        tuner = RandomizedSearchCV(
            GPEmulatorBins,
            param_grid,
            n_iter=n_iter,
            cv=Config.CV_FOLDS,
            scoring='bins_r2',
            random_state=Config.RANDOM_STATE,
            verbose=1
        )
    
    tuner.fit(X, bin_frequencies, other_stats, Config.STAT_NAMES)
    return tuner


def tune_nn_custom(
    X, bin_frequencies, other_stats,
    search_type='random',
    n_iter=20
):
    """
    Custom NN hyperparameter tuning with more control.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    search_type : str
        'grid' or 'random'
    n_iter : int
        Number of random iterations
    
    Returns
    -------
    tuner : fitted tuner
    """
    print("\n" + "="*70)
    print("NEURAL NETWORK HYPERPARAMETER TUNING")
    print("="*70)
    
    # Fixed parameters
    fixed_params = {
        'n_parameters': X.shape[1],
        'n_bins': bin_frequencies.shape[1],
        'n_other_stats': other_stats.shape[1],
        'patience': 20
    }
    
    if search_type == 'grid':
        # Smaller grid for feasibility
        param_grid = {
            **fixed_params,
            'n_hidden_layers': [2, 3],
            'n_nodes': [32, 64],
            'activation': ['relu'],
            'learning_rate': [1e-3, 5e-4],
            'l2_reg': [1e-4, 1e-5],
            'dropout_rate': [0.2, 0.3]
        }
        
        tuner = GridSearchCV(
            NNEmulatorBins,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring='bins_r2',
            random_state=Config.RANDOM_STATE,
            verbose=1
        )
    else:
        # Comprehensive random search
        param_grid = {
            **fixed_params,
            'n_hidden_layers': [2, 3, 4],
            'n_nodes': [16, 32, 64, 128],
            'activation': ['relu', 'leaky_relu'],
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'l2_reg': [1e-5, 1e-4, 1e-3],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4]
        }
        
        tuner = RandomizedSearchCV(
            NNEmulatorBins,
            param_grid,
            n_iter=n_iter,
            cv=Config.CV_FOLDS,
            scoring='bins_r2',
            random_state=Config.RANDOM_STATE,
            verbose=1
        )
    
    tuner.fit(X, bin_frequencies, other_stats, Config.STAT_NAMES)
    return tuner


def tune_both_models(X, bin_frequencies, other_stats, n_iter_nn=20):
    """
    Tune both GP and NN models and compare.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    n_iter_nn : int
        Number of NN configurations to try
    
    Returns
    -------
    results : dict
        Tuning results for both models
    """
    results = {}
    
    # Tune GP (grid search is feasible for GP)
    print("\n" + "="*70)
    print("STEP 1: TUNING GAUSSIAN PROCESS")
    print("="*70)
    gp_tuner = tune_gp_custom(X, bin_frequencies, other_stats, 'grid')
    gp_tuner.print_results(top_n=5)
    results['gp'] = gp_tuner
    
    # Tune NN (random search recommended for NN)
    print("\n" + "="*70)
    print("STEP 2: TUNING NEURAL NETWORK")
    print("="*70)
    nn_tuner = tune_nn_custom(X, bin_frequencies, other_stats, 'random', n_iter_nn)
    nn_tuner.print_results(top_n=5)
    results['nn'] = nn_tuner
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON: BEST GP vs BEST NN")
    print("="*70)
    print(f"\n{'Model':<15} {'Best Score (R²)':<20} {'Best Parameters'}")
    print("-"*70)
    print(f"{'GP':<15} {gp_tuner.best_score_:<20.4f} {gp_tuner.best_params_}")
    print(f"{'NN':<15} {nn_tuner.best_score_:<20.4f} {nn_tuner.best_params_}")
    
    if gp_tuner.best_score_ > nn_tuner.best_score_:
        winner = 'GP'
        margin = gp_tuner.best_score_ - nn_tuner.best_score_
    else:
        winner = 'NN'
        margin = nn_tuner.best_score_ - gp_tuner.best_score_
    
    print(f"\n🏆 Winner: {winner} (by {margin:.4f})")
    print("="*70)
    
    return results


def save_tuning_results(tuner, filepath):
    """
    Save tuning results to file.
    
    Parameters
    ----------
    tuner : HyperparameterTuner
        Fitted tuner
    filepath : str or Path
        Path to save results
    """
    import json
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON
    results_to_save = {
        'best_params': str(tuner.best_params_),
        'best_score': float(tuner.best_score_),
        'scoring_metric': tuner.scoring,
        'cv_folds': tuner.cv,
        'all_results': [
            {
                'params': str(r['params']),
                'score': float(r['score']),
                'cv_results': {k: float(v) for k, v in r['cv_results'].items()}
            }
            for r in tuner.results_
        ]
    }
    
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


def main():
    """Main tuning function."""
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for AMOC emulators'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='both',
        choices=['gp', 'nn', 'both'],
        help='Which model to tune: gp, nn, or both'
    )
    parser.add_argument(
        '--search',
        type=str,
        default='auto',
        choices=['grid', 'random', 'auto'],
        help='Search type (auto=grid for GP, random for NN)'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=20,
        help='Number of iterations for random search'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    Config.display()
    
    # ==========================================
    # Load data
    # ==========================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # TODO: Load your actual data
    print("\nWARNING: Please load your actual data.")
    print("Expected format:")
    print("  - X: (n_samples, n_parameters)")
    print("  - bin_frequencies: (n_samples, n_bins)")
    print("  - other_stats: (n_samples, 4)")
    
    # Uncomment when you have data:
    # X = np.load('data/processed/input_parameters.npy')
    # bin_frequencies = np.load('data/processed/bin_frequencies.npy')
    # other_stats = np.load('data/processed/other_stats.npy')
    
    # ==========================================
    # Perform tuning
    # ==========================================
    # if args.model == 'both':
    #     results = tune_both_models(X, bin_frequencies, other_stats, args.n_iter)
    #     
    #     if args.save:
    #         save_tuning_results(results['gp'], 'tuning_results/gp_tuning.json')
    #         save_tuning_results(results['nn'], 'tuning_results/nn_tuning.json')
    # 
    # elif args.model == 'gp':
    #     search_type = args.search if args.search != 'auto' else 'grid'
    #     tuner = tune_gp_custom(X, bin_frequencies, other_stats, search_type, args.n_iter)
    #     tuner.print_results(top_n=10)
    #     
    #     if args.save:
    #         save_tuning_results(tuner, 'tuning_results/gp_tuning.json')
    # 
    # elif args.model == 'nn':
    #     search_type = args.search if args.search != 'auto' else 'random'
    #     tuner = tune_nn_custom(X, bin_frequencies, other_stats, search_type, args.n_iter)
    #     tuner.print_results(top_n=10)
    #     
    #     if args.save:
    #         save_tuning_results(tuner, 'tuning_results/nn_tuning.json')
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Update config.py with the best parameters")
    print("2. Train final model: python main.py --mode train")
    print("3. Evaluate: python main.py --mode cv")


if __name__ == "__main__":
    main()
