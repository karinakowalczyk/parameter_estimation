"""
Training script for AMOC emulators (GP and NN).

This script handles the complete training pipeline:
- Load and process ensemble data
- Train emulator (GP or NN based on config)
- Evaluate on test set
- Visualize predictions
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

from src.models import GPEmulatorBins, NNEmulatorBins
from src.evaluation import evaluate_model, print_test_results
from src.visualization import plot_prediction_comparison
from config.config import Config


def get_emulator_class(model_type: str):
    """Get the appropriate emulator class based on model type."""
    if model_type == 'gp':
        return GPEmulatorBins
    elif model_type == 'nn':
        return NNEmulatorBins
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_emulator(X, bin_frequencies, other_stats, config=None):
    """
    Train emulator with train/test split.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics (mean, std, q25, q75)
    config : Config, optional
        Configuration object
    
    Returns
    -------
    model : GPEmulatorBins or NNEmulatorBins
        Trained model
    test_data : dict
        Dictionary containing test set data and predictions
    """
    if config is None:
        config = Config
    
    # Split data
    X_train, X_test, bin_train, bin_test, Y_train, Y_test = train_test_split(
        X, bin_frequencies, other_stats, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Input dimensions: {X_train.shape[1]}")
    print(f"Number of bins: {bin_train.shape[1]}")
    print(f"Number of other stats: {Y_train.shape[1]}")
    
    # Get model class and parameters
    EmulatorClass = get_emulator_class(config.MODEL_TYPE)
    
    # Initialize model based on type
    if config.MODEL_TYPE == 'gp':
        print(f"\nTraining GP emulator with kernel: {config.KERNEL}")
        model = EmulatorClass(kernel=config.KERNEL)
        model.fit(X_train, Y_train, bin_train)
        
    elif config.MODEL_TYPE == 'nn':
        print(f"\nTraining NN emulator:")
        print(f"  Hidden layers: {config.NN_N_HIDDEN_LAYERS}")
        print(f"  Nodes per layer: {config.NN_N_NODES}")
        print(f"  Activation: {config.NN_ACTIVATION}")
        print(f"  Learning rate: {config.NN_LEARNING_RATE}")
        
        model = EmulatorClass(
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
        
        model.fit(
            X_train, Y_train, bin_train,
            epochs=config.NN_EPOCHS,
            batch_size=config.NN_BATCH_SIZE,
            validation_split=config.NN_VALIDATION_SPLIT,
            verbose=config.NN_VERBOSE
        )
    
    print("Model training complete!")
    
    # Predict on test set
    print("\nGenerating predictions on test set...")
    prediction_result = model.predict(X_test, return_std=True)
    
    # Handle different return formats (GP returns 4 values, NN returns 4 with None for stds)
    if len(prediction_result) == 4:
        bin_pred, Y_pred, bin_std, Y_std = prediction_result
    else:
        bin_pred, Y_pred = prediction_result
        bin_std, Y_std = None, None
    
    # Store test data
    test_data = {
        'X_test': X_test,
        'bin_test': bin_test,
        'bin_pred': bin_pred,
        'bin_std': bin_std,
        'Y_test': Y_test,
        'Y_pred': Y_pred,
        'Y_std': Y_std
    }
    
    return model, test_data


def main():
    """Main training function."""
    
    # Display configuration
    Config.display()
    
    # ==========================================
    # Load your data here
    # ==========================================
    # TODO: Replace with your actual data loading
    # This is a placeholder - you'll need to load your ensemble data
    
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Example: Load from your processed data
    # X = np.load('data/processed/input_parameters.npy')
    # bin_frequencies = np.load('data/processed/bin_frequencies.npy')
    # other_stats = np.load('data/processed/other_stats.npy')
    
    # For demonstration, create dummy data structure
    print("\nWARNING: Using placeholder data structure.")
    print("Please replace with your actual data loading code.")
    print("\nExpected data format:")
    print("  - X: shape (n_samples, n_parameters)")
    print("  - bin_frequencies: shape (n_samples, n_bins)")
    print("  - other_stats: shape (n_samples, 4)  # mean, std, q25, q75")
    
    # Uncomment and modify when you have real data:
    # from src.data_preparation import process_ensemble_data, prepare_ml_data
    # model_files = list(Config.ENSEMBLE_DIR.glob("*.nc"))
    # ensemble_stats = process_ensemble_data(model_files, n_bins=Config.N_BINS)
    # bin_frequencies, other_stats = prepare_ml_data(ensemble_stats, Config.STAT_NAMES)
    # X = ... # Your input parameters array
    
    # ==========================================
    # Train model
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # model, test_data = train_emulator(X, bin_frequencies, other_stats, Config)
    
    # ==========================================
    # Evaluate model
    # ==========================================
    # print("\n" + "="*60)
    # print("EVALUATING MODEL")
    # print("="*60)
    
    # metrics = evaluate_model(
    #     model,
    #     test_data['X_test'],
    #     test_data['Y_test'],
    #     test_data['bin_test'],
    #     Config.STAT_NAMES
    # )
    
    # print_test_results(metrics)
    
    # ==========================================
    # Visualize predictions
    # ==========================================
    # print("\n" + "="*60)
    # print("VISUALIZING PREDICTIONS")
    # print("="*60)
    
    # # Note: For NN models, bin_std and Y_std will be None
    # if test_data['bin_std'] is not None:
    #     plot_prediction_comparison(
    #         test_data['bin_test'],
    #         test_data['bin_pred'],
    #         test_data['bin_std'],
    #         test_data['Y_test'],
    #         test_data['Y_pred'],
    #         n_bins=Config.N_BINS,
    #         n_samples=Config.N_TEST_SAMPLES_TO_PLOT,
    #         stat_names=Config.STAT_NAMES
    #     )
    # else:
    #     # For NN: plot without uncertainties
    #     from src.visualization import plot_prediction_comparison_no_std
    #     plot_prediction_comparison_no_std(
    #         test_data['bin_test'],
    #         test_data['bin_pred'],
    #         test_data['Y_test'],
    #         test_data['Y_pred'],
    #         n_bins=Config.N_BINS,
    #         n_samples=Config.N_TEST_SAMPLES_TO_PLOT,
    #         stat_names=Config.STAT_NAMES
    #     )
    
    # ==========================================
    # Save model
    # ==========================================
    # if Config.SAVE_MODELS:
    #     Config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #     model_name = f"{Config.MODEL_TYPE}_emulator_bins.pkl"
    #     model_path = Config.MODEL_OUTPUT_DIR / model_name
    #     
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(model, f)
    #     
    #     print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()"""
Training script for AMOC GP emulator.

This script handles the complete training pipeline:
- Load and process ensemble data
- Train GP emulator
- Evaluate on test set
- Visualize predictions
"""

import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

from src.models import GPEmulatorBins
from src.evaluation import evaluate_model, print_test_results
from src.visualization import plot_prediction_comparison
from config.config import Config


def train_emulator(X, bin_frequencies, other_stats, config=None):
    """
    Train GP emulator with train/test split.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics (mean, std, q25, q75)
    config : Config, optional
        Configuration object
    
    Returns
    -------
    model : GPEmulatorBins
        Trained model
    test_data : dict
        Dictionary containing test set data and predictions
    """
    if config is None:
        config = Config
    
    # Split data
    X_train, X_test, bin_train, bin_test, Y_train, Y_test = train_test_split(
        X, bin_frequencies, other_stats, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Input dimensions: {X_train.shape[1]}")
    print(f"Number of bins: {bin_train.shape[1]}")
    print(f"Number of other stats: {Y_train.shape[1]}")
    
    # Initialize and fit model
    print(f"\nTraining GP emulator with kernel: {config.KERNEL}")
    model = GPEmulatorBins(kernel=config.KERNEL)
    model.fit(X_train, Y_train, bin_train)
    print("Model training complete!")
    
    # Predict on test set
    print("\nGenerating predictions on test set...")
    bin_pred, Y_pred, bin_std, Y_std = model.predict(X_test, return_std=True)
    
    # Store test data
    test_data = {
        'X_test': X_test,
        'bin_test': bin_test,
        'bin_pred': bin_pred,
        'bin_std': bin_std,
        'Y_test': Y_test,
        'Y_pred': Y_pred,
        'Y_std': Y_std
    }
    
    return model, test_data


def main():
    """Main training function."""
    
    # Display configuration
    Config.display()
    
    # ==========================================
    # Load your data here
    # ==========================================
    # TODO: Replace with your actual data loading
    # This is a placeholder - you'll need to load your ensemble data
    
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Example: Load from your processed data
    # X = np.load('data/processed/input_parameters.npy')
    # bin_frequencies = np.load('data/processed/bin_frequencies.npy')
    # other_stats = np.load('data/processed/other_stats.npy')
    
    # For demonstration, create dummy data structure
    print("\nWARNING: Using placeholder data structure.")
    print("Please replace with your actual data loading code.")
    print("\nExpected data format:")
    print("  - X: shape (n_samples, n_parameters)")
    print("  - bin_frequencies: shape (n_samples, n_bins)")
    print("  - other_stats: shape (n_samples, 4)  # mean, std, q25, q75")
    
    # Uncomment and modify when you have real data:
    # from src.data_preparation import process_ensemble_data, prepare_ml_data
    # model_files = list(Config.ENSEMBLE_DIR.glob("*.nc"))
    # ensemble_stats = process_ensemble_data(model_files, n_bins=Config.N_BINS)
    # bin_frequencies, other_stats = prepare_ml_data(ensemble_stats, Config.STAT_NAMES)
    # X = ... # Your input parameters array
    
    # ==========================================
    # Train model
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # model, test_data = train_emulator(X, bin_frequencies, other_stats, Config)
    
    # ==========================================
    # Evaluate model
    # ==========================================
    # print("\n" + "="*60)
    # print("EVALUATING MODEL")
    # print("="*60)
    
    # metrics = evaluate_model(
    #     model,
    #     test_data['X_test'],
    #     test_data['Y_test'],
    #     test_data['bin_test'],
    #     Config.STAT_NAMES
    # )
    
    # print_test_results(metrics)
    
    # ==========================================
    # Visualize predictions
    # ==========================================
    # print("\n" + "="*60)
    # print("VISUALIZING PREDICTIONS")
    # print("="*60)
    
    # plot_prediction_comparison(
    #     test_data['bin_test'],
    #     test_data['bin_pred'],
    #     test_data['bin_std'],
    #     test_data['Y_test'],
    #     test_data['Y_pred'],
    #     n_bins=Config.N_BINS,
    #     n_samples=Config.N_TEST_SAMPLES_TO_PLOT,
    #     stat_names=Config.STAT_NAMES
    # )
    
    # ==========================================
    # Save model
    # ==========================================
    # if Config.SAVE_MODELS:
    #     Config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #     model_path = Config.MODEL_OUTPUT_DIR / "gp_emulator_bins.pkl"
    #     
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(model, f)
    #     
    #     print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
