"""
Evaluation script for AMOC emulators (GP and NN).

This script performs cross-validation and model comparison
across different configurations.
"""

import numpy as np
from sklearn.gaussian_process.kernels import Matern

from src.models import GPEmulatorBins, NNEmulatorBins
from src.evaluation import crossvalidation_bins, print_cv_results
from config.config import Config, KernelConfigs, NNConfigs


def get_emulator_class(model_type: str):
    """Get the appropriate emulator class based on model type."""
    if model_type == 'gp':
        return GPEmulatorBins
    elif model_type == 'nn':
        return NNEmulatorBins
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_cross_validation(X, bin_frequencies, other_stats, config=None):
    """
    Run cross-validation with configured model.
    
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
        Cross-validation results
    """
    if config is None:
        config = Config
    
    print(f"Running {config.CV_FOLDS}-fold cross-validation...")
    print(f"Model type: {config.MODEL_TYPE.upper()}")
    
    EmulatorClass = get_emulator_class(config.MODEL_TYPE)
    
    # Get model parameters based on type
    if config.MODEL_TYPE == 'gp':
        print(f"Kernel: {config.KERNEL}")
        emulator_params = {'kernel': config.KERNEL}
    elif config.MODEL_TYPE == 'nn':
        print(f"Architecture: {config.NN_N_HIDDEN_LAYERS} layers x {config.NN_N_NODES} nodes")
        emulator_params = config.get_model_params(
            n_parameters=X.shape[1],
            n_bins=bin_frequencies.shape[1],
            n_other_stats=other_stats.shape[1]
        )
    
    print("=" * 60)
    
    results = crossvalidation_bins(
        X, 
        other_stats, 
        bin_frequencies,
        EmulatorClass,
        emulator_params=emulator_params,
        cv=config.CV_FOLDS,
        output_names=config.STAT_NAMES,
        random_state=config.RANDOM_STATE
    )
    
    return results


def compare_gp_kernels(X, bin_frequencies, other_stats, config=None):
    """
    Compare different GP kernel configurations.
    
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
    comparison : dict
        Dictionary mapping kernel names to their CV results
    """
    if config is None:
        config = Config
    
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT GP KERNELS")
    print("=" * 60)
    
    kernels = {
        'Matern ν=1.5': KernelConfigs.MATERN_1_5,
        'Matern ν=2.5': KernelConfigs.MATERN_2_5,
        'RBF + White': KernelConfigs.RBF_BASIC
    }
    
    comparison = {}
    
    for name, kernel in kernels.items():
        print(f"\nEvaluating {name}...")
        emulator_params = {'kernel': kernel}
        
        results = crossvalidation_bins(
            X,
            other_stats,
            bin_frequencies,
            GPEmulatorBins,
            emulator_params=emulator_params,
            cv=config.CV_FOLDS,
            output_names=config.STAT_NAMES,
            random_state=config.RANDOM_STATE
        )
        
        comparison[name] = results
        print_cv_results(results, title=f"{name} Results")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("KERNEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Kernel':<20} {'Bins RMSE':<15} {'Bins R²':<15} {'Mean RMSE':<15}")
    print("-" * 60)
    
    for name, results in comparison.items():
        print(f"{name:<20} {results['bins_rmse']:<15.4f} "
              f"{results['bins_r2']:<15.3f} {results['mean_rmse']:<15.4f}")
    
    return comparison


def compare_nn_architectures(X, bin_frequencies, other_stats, config=None):
    """
    Compare different NN architecture configurations.
    
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
    comparison : dict
        Dictionary mapping architecture names to their CV results
    """
    if config is None:
        config = Config
    
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT NN ARCHITECTURES")
    print("=" * 60)
    
    architectures = {
        'Small': NNConfigs.SMALL,
        'Medium': NNConfigs.MEDIUM,
        'Large': NNConfigs.LARGE
    }
    
    comparison = {}
    
    for name, arch_params in architectures.items():
        print(f"\nEvaluating {name} architecture...")
        emulator_params = {
            'n_parameters': X.shape[1],
            'n_bins': bin_frequencies.shape[1],
            'n_other_stats': other_stats.shape[1],
            **arch_params,
            'patience': config.NN_PATIENCE
        }
        
        results = crossvalidation_bins(
            X,
            other_stats,
            bin_frequencies,
            NNEmulatorBins,
            emulator_params=emulator_params,
            cv=config.CV_FOLDS,
            output_names=config.STAT_NAMES,
            random_state=config.RANDOM_STATE
        )
        
        comparison[name] = results
        print_cv_results(results, title=f"{name} Architecture Results")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<20} {'Bins RMSE':<15} {'Bins R²':<15} {'Mean RMSE':<15}")
    print("-" * 60)
    
    for name, results in comparison.items():
        print(f"{name:<20} {results['bins_rmse']:<15.4f} "
              f"{results['bins_r2']:<15.3f} {results['mean_rmse']:<15.4f}")
    
    return comparison


def compare_gp_vs_nn(X, bin_frequencies, other_stats, config=None):
    """
    Compare GP and NN model performance.
    
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
    comparison : dict
        Dictionary with results for both model types
    """
    if config is None:
        config = Config
    
    print("\n" + "=" * 60)
    print("COMPARING GP vs NN MODELS")
    print("=" * 60)
    
    comparison = {}
    
    # Evaluate GP
    print("\nEvaluating Gaussian Process...")
    gp_params = {'kernel': config.KERNEL}
    gp_results = crossvalidation_bins(
        X, other_stats, bin_frequencies,
        GPEmulatorBins,
        emulator_params=gp_params,
        cv=config.CV_FOLDS,
        output_names=config.STAT_NAMES,
        random_state=config.RANDOM_STATE
    )
    comparison['GP'] = gp_results
    print_cv_results(gp_results, title="GP Results")
    
    # Evaluate NN
    print("\nEvaluating Neural Network...")
    nn_params = config.get_model_params(
        n_parameters=X.shape[1],
        n_bins=bin_frequencies.shape[1],
        n_other_stats=other_stats.shape[1]
    )
    nn_results = crossvalidation_bins(
        X, other_stats, bin_frequencies,
        NNEmulatorBins,
        emulator_params=nn_params,
        cv=config.CV_FOLDS,
        output_names=config.STAT_NAMES,
        random_state=config.RANDOM_STATE
    )
    comparison['NN'] = nn_results
    print_cv_results(nn_results, title="NN Results")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("GP vs NN COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Bins RMSE':<15} {'Bins R²':<15} {'Mean RMSE':<15}")
    print("-" * 60)
    
    for name, results in comparison.items():
        print(f"{name:<20} {results['bins_rmse']:<15.4f} "
              f"{results['bins_r2']:<15.3f} {results['mean_rmse']:<15.4f}")
    
    return comparison


def main():
    """Main evaluation function."""
    
    # Display configuration
    Config.display()
    
    # ==========================================
    # Load your data here
    # ==========================================
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # TODO: Replace with your actual data loading
    print("\nWARNING: Using placeholder data structure.")
    print("Please replace with your actual data loading code.")
    
    # Example:
    # X = np.load('data/processed/input_parameters.npy')
    # bin_frequencies = np.load('data/processed/bin_frequencies.npy')
    # other_stats = np.load('data/processed/other_stats.npy')
    
    # ==========================================
    # Run single cross-validation
    # ==========================================
    print("\n" + "="*60)
    print("CROSS-VALIDATION WITH CONFIGURED MODEL")
    print("="*60)
    
    # results = run_cross_validation(X, bin_frequencies, other_stats, Config)
    # print_cv_results(results)
    
    # ==========================================
    # Compare configurations (optional)
    # ==========================================
    # Uncomment to compare different configurations
    
    # Compare GP kernels
    # comparison_gp = compare_gp_kernels(X, bin_frequencies, other_stats, Config)
    
    # Compare NN architectures
    # comparison_nn = compare_nn_architectures(X, bin_frequencies, other_stats, Config)
    
    # Compare GP vs NN
    # comparison_models = compare_gp_vs_nn(X, bin_frequencies, other_stats, Config)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
