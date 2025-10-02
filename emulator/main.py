"""
Main entry point for AMOC ML tuning project.

This script orchestrates the complete pipeline from data loading
to model training and evaluation.
"""

import numpy as np
from pathlib import Path
import argparse

from src.data_preparation import (
    load_amoc_data, 
    process_ensemble_data, 
    prepare_ml_data
)
from src.visualization import plot_amoc_ensemble_simple
from experiments.train_model import train_emulator
from experiments.evaluate_model import run_cross_validation, compare_kernels
from src.evaluation import print_cv_results, evaluate_model, print_test_results
from config.config import Config


def load_and_prepare_data(default_file, ensemble_files, config):
    """
    Load and prepare all data for ML model.
    
    Parameters
    ----------
    default_file : str or Path
        Path to default run file
    ensemble_files : list of str
        List of ensemble run files
    config : Config
        Configuration object
    
    Returns
    -------
    data : dict
        Dictionary containing all prepared data
    """
    print("\n" + "="*60)
    print("LOADING AND PROCESSING DATA")
    print("="*60)
    
    # Load default run
    print(f"\nLoading default run from: {default_file}")
    amoc_default, time_default = load_amoc_data(default_file)
    
    from src.data_preparation import compute_simple_summary_stats
    default_stats = compute_simple_summary_stats(
        amoc_default, time_default, n_bins=config.N_BINS
    )
    print(f"Default run: {len(amoc_default)} time steps")
    
    # Process ensemble runs
    print(f"\nProcessing {len(ensemble_files)} ensemble runs...")
    ensemble_stats = process_ensemble_data(
        ensemble_files, 
        remove_spinup=config.REMOVE_SPINUP,
        n_bins=config.N_BINS
    )
    
    # Prepare ML data
    print("\nPreparing ML data...")
    bin_frequencies, other_stats = prepare_ml_data(
        ensemble_stats, 
        config.STAT_NAMES
    )
    
    print(f"\nData prepared:")
    print(f"  - Bin frequencies shape: {bin_frequencies.shape}")
    print(f"  - Other stats shape: {other_stats.shape}")
    print(f"  - Number of samples: {len(ensemble_stats)}")
    
    data = {
        'amoc_default': amoc_default,
        'default_stats': default_stats,
        'ensemble_stats': ensemble_stats,
        'ensemble_files': ensemble_files,
        'bin_frequencies': bin_frequencies,
        'other_stats': other_stats
    }
    
    return data


def visualize_data(data, config):
    """
    Visualize ensemble data.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_and_prepare_data
    config : Config
        Configuration object
    """
    print("\n" + "="*60)
    print("VISUALIZING ENSEMBLE DATA")
    print("="*60)
    
    plot_amoc_ensemble_simple(
        data['ensemble_stats'],
        data['ensemble_files'],
        data['amoc_default'],
        data['default_stats'],
        n_runs=config.N_RUNS_TO_PLOT,
        figsize_per_run=config.FIGSIZE_PER_RUN,
        n_bins=config.N_BINS
    )


def run_pipeline(X, data, mode='train', config=None):
    """
    Run the complete ML pipeline.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    data : dict
        Data dictionary containing bin frequencies and statistics
    mode : str, default='train'
        Pipeline mode: 'train', 'cv', or 'compare'
    config : Config, optional
        Configuration object
    
    Returns
    -------
    results : dict
        Pipeline results
    """
    if config is None:
        config = Config
    
    bin_frequencies = data['bin_frequencies']
    other_stats = data['other_stats']
    
    if mode == 'train':
        # Train and evaluate model
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        model, test_data = train_emulator(X, bin_frequencies, other_stats, config)
        
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        metrics = evaluate_model(
            model,
            test_data['X_test'],
            test_data['Y_test'],
            test_data['bin_test'],
            config.STAT_NAMES
        )
        
        print_test_results(metrics)
        
        results = {'model': model, 'test_data': test_data, 'metrics': metrics}
        
    elif mode == 'cv':
        # Cross-validation
        print("\n" + "="*60)
        print("CROSS-VALIDATION")
        print("="*60)
        
        cv_results = run_cross_validation(X, bin_frequencies, other_stats, config)
        print_cv_results(cv_results)
        
        results = {'cv_results': cv_results}
        
    elif mode == 'compare':
        # Compare kernels
        comparison = compare_kernels(X, bin_frequencies, other_stats, config)
        results = {'comparison': comparison}
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return results


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='AMOC ML Tuning Pipeline')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train',
        choices=['train', 'cv', 'compare'],
        help='Pipeline mode: train, cv (cross-validation), or compare (kernel comparison)'
    )
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Visualize ensemble data'
    )
    
    args = parser.parse_args()
    
    # Display configuration
    Config.display()
    
    # ==========================================
    # IMPORTANT: Load your data here
    # ==========================================
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)
    print("\nWARNING: You need to specify your data files and input parameters.")
    print("\nPlease modify this section with:")
    print("  1. Path to your default run file")
    print("  2. List of ensemble run files")
    print("  3. Your input parameter array X")
    print("\nExample:")
    print("  default_file = '../default_long_run/ocn_ts.nc'")
    print("  ensemble_files = list(Path('../ensemble_runs').glob('*.nc'))")
    print("  X = np.load('input_parameters.npy')  # Your parameter values")
    
    # TODO: Uncomment and modify these lines with your actual data
    # default_file = Config.DEFAULT_RUN_FILE
    # ensemble_files = list(Config.ENSEMBLE_DIR.glob("*.nc"))
    # 
    # data = load_and_prepare_data(default_file, ensemble_files, Config)
    # 
    # # Load your input parameters
    # X = ...  # Your input parameter array
    # 
    # # Optionally visualize data
    # if args.visualize:
    #     visualize_data(data, Config)
    # 
    # # Run pipeline
    # results = run_pipeline(X, data, mode=args.mode, config=Config)
    # 
    # print("\n" + "="*60)
    # print("PIPELINE COMPLETE")
    # print("="*60)


if __name__ == "__main__":
    main()
