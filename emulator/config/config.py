"""
Configuration file for AMOC ML tuning project.

Contains hyperparameters, paths, and settings for data processing,
model training, and evaluation.
"""

from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from pathlib import Path


class Config:
    """Main configuration class for the project."""
    
    # ==========================================
    # Data Configuration
    # ==========================================
    # Paths
    DATA_DIR = Path("../data")
    DEFAULT_RUN_FILE = DATA_DIR / "default_long_run" / "ocn_ts.nc"
    ENSEMBLE_DIR = DATA_DIR / "ensemble_runs"
    
    # Data processing
    N_BINS = 10
    REMOVE_SPINUP = False
    SPINUP_FRACTION = 0.1
    
    # Summary statistics to compute
    STAT_NAMES = ['mean', 'std', 'q25', 'q75']
    
    # ==========================================
    # Model Selection
    # ==========================================
    # Choose model type: 'gp' or 'nn'
    MODEL_TYPE = 'gp'  # 'gp' for Gaussian Process, 'nn' for Neural Network
    
    # ==========================================
    # GP Model Configuration
    # ==========================================
    # Kernel options - choose one or experiment with different kernels
    
    # Option 1: Matern kernel (recommended for smooth but non-infinitely differentiable functions)
    KERNEL = Matern(length_scale=1.0, nu=1.5)
    
    # Option 2: RBF kernel (for infinitely smooth functions)
    # KERNEL = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
    
    # GP model parameters
    GP_N_RESTARTS = 3
    GP_NORMALIZE_Y = True
    
    # ==========================================
    # Neural Network Configuration
    # ==========================================
    NN_N_HIDDEN_LAYERS = 2
    NN_N_NODES = 32
    NN_ACTIVATION = 'relu'  # 'relu' or 'leaky_relu'
    NN_LEARNING_RATE = 1e-3
    NN_L2_REG = 1e-4
    NN_DROPOUT_RATE = 0.2
    NN_PATIENCE = 20
    NN_EPOCHS = 100
    NN_BATCH_SIZE = 8
    NN_VALIDATION_SPLIT = 0.2
    NN_VERBOSE = 0  # 0=silent, 1=progress bar, 2=one line per epoch
    
    # ==========================================
    # Training Configuration
    # ==========================================
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Cross-validation
    CV_FOLDS = 5
    
    # ==========================================
    # Visualization Configuration
    # ==========================================
    FIGSIZE_PER_RUN = 3
    N_RUNS_TO_PLOT = 10
    N_TEST_SAMPLES_TO_PLOT = 3
    
    # Plot settings
    FIG_DPI = 100
    SAVE_FIGS = False
    FIG_OUTPUT_DIR = Path("figures")
    
    # ==========================================
    # Experiment Tracking
    # ==========================================
    SAVE_MODELS = True
    MODEL_OUTPUT_DIR = Path("saved_models")
    EXPERIMENT_LOG_FILE = Path("experiments") / "experiment_log.csv"
    
    @classmethod
    def display(cls):
        """Display all configuration settings."""
        print("=" * 60)
        print("AMOC ML Tuning Configuration")
        print("=" * 60)
        print("\nData Configuration:")
        print(f"  N_BINS: {cls.N_BINS}")
        print(f"  REMOVE_SPINUP: {cls.REMOVE_SPINUP}")
        print(f"  STAT_NAMES: {cls.STAT_NAMES}")
        
        print(f"\nModel Type: {cls.MODEL_TYPE.upper()}")
        
        if cls.MODEL_TYPE == 'gp':
            print("\nGP Model Configuration:")
            print(f"  KERNEL: {cls.KERNEL}")
            print(f"  GP_N_RESTARTS: {cls.GP_N_RESTARTS}")
        elif cls.MODEL_TYPE == 'nn':
            print("\nNeural Network Configuration:")
            print(f"  N_HIDDEN_LAYERS: {cls.NN_N_HIDDEN_LAYERS}")
            print(f"  N_NODES: {cls.NN_N_NODES}")
            print(f"  ACTIVATION: {cls.NN_ACTIVATION}")
            print(f"  LEARNING_RATE: {cls.NN_LEARNING_RATE}")
            print(f"  L2_REG: {cls.NN_L2_REG}")
            print(f"  DROPOUT_RATE: {cls.NN_DROPOUT_RATE}")
            print(f"  EPOCHS: {cls.NN_EPOCHS}")
            print(f"  BATCH_SIZE: {cls.NN_BATCH_SIZE}")
        
        print("\nTraining Configuration:")
        print(f"  TEST_SIZE: {cls.TEST_SIZE}")
        print(f"  CV_FOLDS: {cls.CV_FOLDS}")
        print(f"  RANDOM_STATE: {cls.RANDOM_STATE}")
        
        print("\n" + "=" * 60)
    
    @classmethod
    def get_model_params(cls, n_parameters=None, n_bins=None, n_other_stats=None):
        """
        Get model parameters based on MODEL_TYPE.
        
        Parameters
        ----------
        n_parameters : int, optional
            Number of input parameters (required for NN)
        n_bins : int, optional
            Number of bins (required for NN)
        n_other_stats : int, optional
            Number of other statistics (required for NN)
        
        Returns
        -------
        params : dict
            Dictionary of model parameters
        """
        if cls.MODEL_TYPE == 'gp':
            return {'kernel': cls.KERNEL}
        elif cls.MODEL_TYPE == 'nn':
            if any(x is None for x in [n_parameters, n_bins, n_other_stats]):
                raise ValueError(
                    "n_parameters, n_bins, and n_other_stats are required for NN"
                )
            return {
                'n_parameters': n_parameters,
                'n_bins': n_bins,
                'n_other_stats': n_other_stats,
                'n_hidden_layers': cls.NN_N_HIDDEN_LAYERS,
                'n_nodes': cls.NN_N_NODES,
                'activation': cls.NN_ACTIVATION,
                'learning_rate': cls.NN_LEARNING_RATE,
                'l2_reg': cls.NN_L2_REG,
                'dropout_rate': cls.NN_DROPOUT_RATE,
                'patience': cls.NN_PATIENCE
            }
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {cls.MODEL_TYPE}")


# Alternative kernel configurations for experimentation
class KernelConfigs:
    """Pre-defined kernel configurations for easy experimentation."""
    
    MATERN_1_5 = Matern(length_scale=1.0, nu=1.5)
    MATERN_2_5 = Matern(length_scale=1.0, nu=2.5)
    RBF_BASIC = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
    RBF_ANISOTROPIC = RBF(length_scale=[1.0, 1.0, 1.0])  # Adjust dimensions as needed


# Neural Network architecture configurations
class NNConfigs:
    """Pre-defined NN configurations for easy experimentation."""
    
    SMALL = {
        'n_hidden_layers': 2,
        'n_nodes': 32,
        'activation': 'relu',
        'learning_rate': 1e-3,
        'l2_reg': 1e-4,
        'dropout_rate': 0.2
    }
    
    MEDIUM = {
        'n_hidden_layers': 3,
        'n_nodes': 64,
        'activation': 'relu',
        'learning_rate': 1e-3,
        'l2_reg': 1e-4,
        'dropout_rate': 0.3
    }
    
    LARGE = {
        'n_hidden_layers': 4,
        'n_nodes': 128,
        'activation': 'leaky_relu',
        'learning_rate': 5e-4,
        'l2_reg': 1e-5,
        'dropout_rate': 0.3
    }
