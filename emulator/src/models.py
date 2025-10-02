"""
Machine learning models for AMOC emulation.

This module contains emulator classes (GP and NN) for predicting AMOC 
bin frequencies and summary statistics.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class GPEmulatorBins:
    """
    Gaussian Process emulator that predicts bin frequencies and summary statistics.
    
    This emulator uses a single multi-output GP to capture correlations between
    bin frequencies and other summary statistics (mean, std, quartiles).
    
    Attributes
    ----------
    kernel : GP kernel
        Kernel function for the Gaussian Process
    scaler_X : sklearn scaler
        Scaler for input features
    scaler_y : sklearn scaler
        Scaler for output features
    gp_model : GaussianProcessRegressor
        Fitted GP model
    is_fitted : bool
        Whether the model has been fitted
    n_bins : int
        Number of histogram bins
    n_other_stats : int
        Number of other summary statistics
    """
    
    def __init__(self, kernel=None, scaler_X=None, scaler_y=None):
        """
        Initialize GP emulator.
        
        Parameters
        ----------
        kernel : GP kernel, optional
            Kernel for the GP. If None, uses RBF + WhiteKernel
        scaler_X : scaler, optional
            Scaler for input features. If None, uses StandardScaler
        scaler_y : scaler, optional
            Scaler for output features. If None, uses StandardScaler
        """
        self.kernel = kernel if kernel is not None else RBF(1.0) + WhiteKernel(1e-6)
        self.scaler_X = scaler_X if scaler_X is not None else StandardScaler()
        self.scaler_y = scaler_y if scaler_y is not None else StandardScaler()
        self.gp_model = None
        self.is_fitted = False
        self.n_bins = None
        self.n_other_stats = None
        
    def fit(self, X, Y, bin_frequencies):
        """
        Fit the GP emulator.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_parameters)
            Input parameters
        Y : array-like, shape (n_samples, n_other_summary_stats)
            Other summary statistics (mean, std, q25, q75)
        bin_frequencies : array-like, shape (n_samples, n_bins)
            Bin frequencies for each sample
        
        Returns
        -------
        self : GPEmulatorBins
            Fitted emulator
        """
        # Store dimensions
        self.n_bins = bin_frequencies.shape[1]
        self.n_other_stats = Y.shape[1]
        
        # Combine bin frequencies and other summary stats
        Y_full = np.hstack([bin_frequencies, Y])
        
        # Scale inputs and outputs
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y_full)
        
        # Fit single multi-output GP (captures correlations between outputs)
        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel, 
            normalize_y=True,
            n_restarts_optimizer=3
        )
        self.gp_model.fit(X_scaled, Y_scaled)
        self.is_fitted = True
        
        return self
        
    def predict(self, X_new, return_std=False):
        """
        Predict bin frequencies and summary stats for new inputs.
        
        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_parameters)
            New input parameters
        return_std : bool, default=False
            Whether to return prediction uncertainties
        
        Returns
        -------
        bin_frequencies_pred : np.ndarray, shape (n_samples, n_bins)
            Predicted bin frequencies
        Y_pred : np.ndarray, shape (n_samples, n_other_summary_stats)
            Predicted other summary statistics
        bin_frequencies_std : np.ndarray, optional, shape (n_samples, n_bins)
            Uncertainty in bin frequencies (if return_std=True)
        Y_std : np.ndarray, optional, shape (n_samples, n_other_summary_stats)
            Uncertainty in other statistics (if return_std=True)
        """
        if not self.is_fitted:
            raise RuntimeError("GPEmulatorBins must be fitted first.")
        
        X_scaled = self.scaler_X.transform(X_new)
        
        # Predict with GP
        if return_std:
            mean_scaled, std_scaled = self.gp_model.predict(X_scaled, return_std=True)
        else:
            mean_scaled = self.gp_model.predict(X_scaled)
            std_scaled = None
        
        # Inverse scaling
        mean_full = self.scaler_y.inverse_transform(mean_scaled)
        
        if return_std and std_scaled is not None:
            std_full = std_scaled * self.scaler_y.scale_
        else:
            std_full = None
        
        # Split bin frequencies and other summary stats
        bin_frequencies_pred = mean_full[:, :self.n_bins]
        Y_pred = mean_full[:, self.n_bins:]
        
        # Ensure bin frequencies are non-negative and sum to 1
        bin_frequencies_pred = self._normalize_bin_frequencies(bin_frequencies_pred)
        
        if return_std and std_full is not None:
            bin_frequencies_std = std_full[:, :self.n_bins]
            Y_std = std_full[:, self.n_bins:]
            
            return bin_frequencies_pred, Y_pred, bin_frequencies_std, Y_std
        else:
            return bin_frequencies_pred, Y_pred
    
    def _normalize_bin_frequencies(self, bin_frequencies):
        """
        Ensure bin frequencies are non-negative and sum to 1.
        
        Parameters
        ----------
        bin_frequencies : np.ndarray
            Raw bin frequency predictions
        
        Returns
        -------
        np.ndarray
            Normalized bin frequencies
        """
        # Clip negative values to 0
        bin_frequencies = np.clip(bin_frequencies, 0, None)
        
        # Normalize each row to sum to 1
        row_sums = bin_frequencies.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        bin_frequencies = bin_frequencies / row_sums
        
        return bin_frequencies
    
    def score(self, X_test, Y_test, bin_frequencies_test):
        """
        Compute R² scores for model predictions.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_parameters)
            Test input parameters
        Y_test : array-like, shape (n_samples, n_other_summary_stats)
            True summary statistics
        bin_frequencies_test : array-like, shape (n_samples, n_bins)
            True bin frequencies
        
        Returns
        -------
        scores : dict
            Dictionary containing:
            - bin_frequencies_r2: R² for bin frequencies
            - other_stats_r2: R² for other statistics
            - overall_r2: Combined R² score
        """
        if not self.is_fitted:
            raise RuntimeError("GPEmulatorBins must be fitted first.")
        
        bin_frequencies_pred, Y_pred = self.predict(X_test, return_std=False)
        
        scores = {
            'bin_frequencies_r2': r2_score(bin_frequencies_test, bin_frequencies_pred),
            'other_stats_r2': r2_score(Y_test, Y_pred),
            'overall_r2': r2_score(
                np.hstack([bin_frequencies_test, Y_test]), 
                np.hstack([bin_frequencies_pred, Y_pred])
            )
        }
        
        return scores


class NNEmulatorBins:
    """
    Neural Network emulator that predicts bin frequencies and summary statistics.
    
    Uses a feed-forward neural network with L2 regularization, dropout,
    and early stopping to prevent overfitting on small datasets.
    
    Attributes
    ----------
    n_parameters : int
        Number of input parameters
    n_bins : int
        Number of histogram bins
    n_other_stats : int
        Number of other summary statistics
    model : keras.Model
        Compiled neural network model
    scaler_X : StandardScaler
        Scaler for input features
    scaler_y : StandardScaler
        Scaler for output features
    is_fitted : bool
        Whether the model has been fitted
    """
    
    def __init__(
        self,
        n_parameters: int,
        n_bins: int,
        n_other_stats: int,
        n_hidden_layers: int = 2,
        n_nodes: int = 32,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-4,
        dropout_rate: float = 0.2,
        patience: int = 20
    ):
        """
        Initialize Neural Network emulator.
        
        Parameters
        ----------
        n_parameters : int
            Number of input parameters
        n_bins : int
            Number of bin frequencies to predict
        n_other_stats : int
            Number of additional summary statistics
        n_hidden_layers : int, default=2
            Number of hidden layers (smaller for small datasets)
        n_nodes : int, default=32
            Nodes per hidden layer
        activation : str, default='relu'
            Activation function: 'relu' or 'leaky_relu'
        learning_rate : float, default=1e-3
            Learning rate for Adam optimizer
        l2_reg : float, default=1e-4
            L2 weight decay for regularization
        dropout_rate : float, default=0.2
            Dropout rate between layers
        patience : int, default=20
            Early stopping patience
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for NNEmulatorBins. "
                "Install it with: pip install tensorflow"
            )
        
        self.n_parameters = n_parameters
        self.n_bins = n_bins
        self.n_other_stats = n_other_stats
        self.output_dim = n_bins + n_other_stats
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.patience = patience
        
        self.model = self._build_model(
            n_hidden_layers, n_nodes, activation,
            learning_rate, l2_reg, dropout_rate
        )
    
    def _build_model(
        self,
        n_hidden_layers: int,
        n_nodes: int,
        activation: str,
        learning_rate: float,
        l2_reg: float,
        dropout_rate: float
    ):
        """
        Build the neural network architecture.
        
        Parameters
        ----------
        n_hidden_layers : int
            Number of hidden layers
        n_nodes : int
            Nodes per hidden layer
        activation : str
            Activation function name
        learning_rate : float
            Learning rate
        l2_reg : float
            L2 regularization strength
        dropout_rate : float
            Dropout rate
        
        Returns
        -------
        model : keras.Model
            Compiled neural network model
        """
        inputs = keras.Input(shape=(self.n_parameters,))
        x = inputs
        
        # Choose activation function
        if activation.lower() == "relu":
            act_fn = layers.ReLU()
        elif activation.lower() in ["leakyrelu", "leaky_relu"]:
            act_fn = layers.LeakyReLU(alpha=0.01)
        else:
            raise ValueError("activation must be 'relu' or 'leaky_relu'")
        
        # Hidden layers with L2 regularization and Dropout
        for _ in range(n_hidden_layers):
            x = layers.Dense(
                n_nodes,
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            )(x)
            x = act_fn(x)
            x = layers.Dropout(dropout_rate)(x)
        
        # Output layer (no activation for regression)
        outputs = layers.Dense(self.output_dim)(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="mse"
        )
        
        return model
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        bin_frequencies: np.ndarray,
        epochs: int = 50,
        batch_size: int = 8,
        validation_split: float = 0.2,
        verbose: int = 0
    ):
        """
        Train the neural network with built-in early stopping.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_parameters)
            Input parameters
        Y : np.ndarray, shape (n_samples, n_other_stats)
            Other summary statistics
        bin_frequencies : np.ndarray, shape (n_samples, n_bins)
            Bin frequencies for each sample
        epochs : int, default=50
            Maximum number of training epochs
        batch_size : int, default=8
            Batch size for training
        validation_split : float, default=0.2
            Fraction of training data to use for validation
        verbose : int, default=0
            Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        
        Returns
        -------
        self : NNEmulatorBins
            Fitted emulator
        """
        # Combine bin frequencies and other stats
        Y_full = np.hstack([bin_frequencies, Y])
        
        # Scale input/output
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y_full)
        
        # Early stopping callback
        es_callback = keras.callbacks.EarlyStopping(
            patience=self.patience,
            restore_best_weights=True
        )
        
        # Train the model
        self.model.fit(
            X_scaled, Y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[es_callback],
            verbose=verbose
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X_new: np.ndarray, return_std: bool = False):
        """
        Predict bin frequencies and summary stats for new inputs.
        
        Note: Neural networks do not provide uncertainties by default,
        so return_std is included for API compatibility but ignored.
        
        Parameters
        ----------
        X_new : np.ndarray, shape (n_samples, n_parameters)
            New input parameters
        return_std : bool, default=False
            Ignored for NN (included for API compatibility)
        
        Returns
        -------
        bin_frequencies_pred : np.ndarray, shape (n_samples, n_bins)
            Predicted bin frequencies
        Y_pred : np.ndarray, shape (n_samples, n_other_stats)
            Predicted other summary statistics
        """
        if not self.is_fitted:
            raise RuntimeError("NNEmulatorBins must be fitted first.")
        
        # Scale inputs and predict
        X_scaled = self.scaler_X.transform(X_new)
        preds_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse scale predictions
        preds = self.scaler_y.inverse_transform(preds_scaled)
        
        # Split predictions
        bin_frequencies_pred = preds[:, :self.n_bins]
        Y_pred = preds[:, self.n_bins:]
        
        # Normalize bin frequencies
        bin_frequencies_pred = self._normalize_bin_frequencies(bin_frequencies_pred)
        
        if return_std:
            # Return None for uncertainties (NN doesn't provide them)
            return bin_frequencies_pred, Y_pred, None, None
        else:
            return bin_frequencies_pred, Y_pred
    
    def _normalize_bin_frequencies(self, bin_frequencies: np.ndarray) -> np.ndarray:
        """
        Ensure bin frequencies are non-negative and sum to 1.
        
        Parameters
        ----------
        bin_frequencies : np.ndarray
            Raw bin frequency predictions
        
        Returns
        -------
        np.ndarray
            Normalized bin frequencies
        """
        # Clip negative values to 0
        bin_frequencies = np.clip(bin_frequencies, 0, None)
        
        # Normalize each row to sum to 1
        row_sums = bin_frequencies.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        bin_frequencies = bin_frequencies / row_sums
        
        return bin_frequencies
    
    def score(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        bin_frequencies_test: np.ndarray
    ) -> dict:
        """
        Compute R² scores for model predictions.
        
        Parameters
        ----------
        X_test : np.ndarray, shape (n_samples, n_parameters)
            Test input parameters
        Y_test : np.ndarray, shape (n_samples, n_other_stats)
            True summary statistics
        bin_frequencies_test : np.ndarray, shape (n_samples, n_bins)
            True bin frequencies
        
        Returns
        -------
        scores : dict
            Dictionary containing:
            - bin_frequencies_r2: R² for bin frequencies
            - other_stats_r2: R² for other statistics
            - overall_r2: Combined R² score
        """
        if not self.is_fitted:
            raise RuntimeError("NNEmulatorBins must be fitted first.")
        
        bin_frequencies_pred, Y_pred = self.predict(X_test)
        
        scores = {
            'bin_frequencies_r2': r2_score(bin_frequencies_test, bin_frequencies_pred),
            'other_stats_r2': r2_score(Y_test, Y_pred),
            'overall_r2': r2_score(
                np.hstack([bin_frequencies_test, Y_test]),
                np.hstack([bin_frequencies_pred, Y_pred])
            )
        }
        
        return scores
