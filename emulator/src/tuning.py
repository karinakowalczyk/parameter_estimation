"""
Hyperparameter tuning module for AMOC emulators.

This module provides GridSearch and RandomizedSearch capabilities
for both GP and NN emulators.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from itertools import product
import time
from typing import Dict, List, Any, Type
import warnings

from src.models import GPEmulatorBins, NNEmulatorBins
from src.evaluation import crossvalidation_bins


class HyperparameterTuner:
    """
    Base class for hyperparameter tuning.
    
    Supports both GridSearch and RandomizedSearch for GP and NN emulators.
    """
    
    def __init__(
        self,
        emulator_class: Type,
        param_grid: Dict[str, List],
        cv: int = 5,
        scoring: str = 'bins_r2',
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize hyperparameter tuner.
        
        Parameters
        ----------
        emulator_class : class
            Emulator class (GPEmulatorBins or NNEmulatorBins)
        param_grid : dict
            Dictionary with parameter names as keys and lists of values to try
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='bins_r2'
            Metric to optimize: 'bins_r2', 'bins_rmse', 'mean_rmse', etc.
        random_state : int, default=42
            Random state for reproducibility
        verbose : int, default=1
            Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.emulator_class = emulator_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        
        self.results_ = None
        self.best_params_ = None
        self.best_score_ = None
        
    def _evaluate_params(
        self,
        params: Dict,
        X: np.ndarray,
        bin_frequencies: np.ndarray,
        other_stats: np.ndarray,
        stat_names: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a single parameter configuration using cross-validation.
        
        Parameters
        ----------
        params : dict
            Parameter configuration to evaluate
        X : np.ndarray
            Input parameters
        bin_frequencies : np.ndarray
            Bin frequency data
        other_stats : np.ndarray
            Summary statistics
        stat_names : list of str
            Names of statistics
        
        Returns
        -------
        results : dict
            Cross-validation results
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            results = crossvalidation_bins(
                X, other_stats, bin_frequencies,
                self.emulator_class,
                emulator_params=params,
                cv=self.cv,
                output_names=stat_names,
                random_state=self.random_state
            )
        
        return results
    
    def _get_score(self, results: Dict[str, float]) -> float:
        """
        Extract the score to optimize from CV results.
        
        Parameters
        ----------
        results : dict
            Cross-validation results
        
        Returns
        -------
        score : float
            Score value (higher is better for R², lower is better for RMSE)
        """
        score = results.get(self.scoring, None)
        if score is None:
            raise ValueError(f"Scoring metric '{self.scoring}' not found in results")
        
        # For RMSE metrics, negate so higher is better
        if 'rmse' in self.scoring.lower() or 'error' in self.scoring.lower():
            score = -score
        
        return score
    
    def fit(
        self,
        X: np.ndarray,
        bin_frequencies: np.ndarray,
        other_stats: np.ndarray,
        stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
    ):
        """
        Perform hyperparameter search (to be implemented by subclasses).
        
        Parameters
        ----------
        X : np.ndarray
            Input parameters
        bin_frequencies : np.ndarray
            Bin frequency data
        other_stats : np.ndarray
            Summary statistics
        stat_names : list of str
            Names of statistics
        
        Returns
        -------
        self : HyperparameterTuner
            Fitted tuner
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def print_results(self, top_n: int = 5):
        """
        Print top N parameter configurations.
        
        Parameters
        ----------
        top_n : int, default=5
            Number of top configurations to display
        """
        if self.results_ is None:
            print("No results available. Run fit() first.")
            return
        
        print("\n" + "="*80)
        print(f"HYPERPARAMETER TUNING RESULTS (Top {top_n})")
        print("="*80)
        print(f"Best Score ({self.scoring}): {self.best_score_:.4f}")
        print(f"Best Parameters: {self.best_params_}")
        print("\n" + "-"*80)
        print(f"{'Rank':<6} {'Score':<12} {'Parameters'}")
        print("-"*80)
        
        for i, result in enumerate(self.results_[:top_n]):
            print(f"{i+1:<6} {result['score']:<12.4f} {result['params']}")
        
        print("="*80 + "\n")


class GridSearchCV(HyperparameterTuner):
    """
    Exhaustive search over specified parameter values.
    
    Tries all possible combinations of parameters in param_grid.
    """
    
    def fit(
        self,
        X: np.ndarray,
        bin_frequencies: np.ndarray,
        other_stats: np.ndarray,
        stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
    ):
        """
        Perform exhaustive grid search.
        
        Parameters
        ----------
        X : np.ndarray
            Input parameters
        bin_frequencies : np.ndarray
            Bin frequency data
        other_stats : np.ndarray
            Summary statistics
        stat_names : list of str
            Names of statistics
        
        Returns
        -------
        self : GridSearchCV
            Fitted tuner with results
        """
        # Generate all parameter combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        n_combinations = len(param_combinations)
        
        if self.verbose > 0:
            print(f"\nGrid Search: Testing {n_combinations} parameter combinations")
            print(f"CV Folds: {self.cv}")
            print(f"Scoring: {self.scoring}")
            print("-"*60)
        
        results = []
        start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            if self.verbose > 0:
                print(f"[{i+1}/{n_combinations}] Testing: {params}")
            
            cv_results = self._evaluate_params(
                params, X, bin_frequencies, other_stats, stat_names
            )
            score = self._get_score(cv_results)
            
            results.append({
                'params': params,
                'score': score,
                'cv_results': cv_results
            })
            
            if self.verbose > 1:
                print(f"  Score: {score:.4f}")
        
        elapsed = time.time() - start_time
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        self.results_ = results
        self.best_params_ = results[0]['params']
        self.best_score_ = results[0]['score']
        
        if self.verbose > 0:
            print(f"\nGrid Search Complete!")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best params: {self.best_params_}")
        
        return self


class RandomizedSearchCV(HyperparameterTuner):
    """
    Randomized search over specified parameter distributions.
    
    Samples n_iter parameter configurations randomly from param_distributions.
    """
    
    def __init__(
        self,
        emulator_class: Type,
        param_distributions: Dict[str, List],
        n_iter: int = 10,
        cv: int = 5,
        scoring: str = 'bins_r2',
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize randomized search.
        
        Parameters
        ----------
        emulator_class : class
            Emulator class
        param_distributions : dict
            Dictionary with parameter names and distributions/lists
        n_iter : int, default=10
            Number of parameter settings to sample
        cv : int, default=5
            Number of CV folds
        scoring : str, default='bins_r2'
            Metric to optimize
        random_state : int, default=42
            Random state
        verbose : int, default=1
            Verbosity level
        """
        super().__init__(
            emulator_class, param_distributions, cv, scoring, random_state, verbose
        )
        self.n_iter = n_iter
        self.rng = np.random.RandomState(random_state)
    
    def _sample_params(self) -> Dict[str, Any]:
        """
        Sample a random parameter configuration.
        
        Returns
        -------
        params : dict
            Sampled parameter configuration
        """
        params = {}
        for key, values in self.param_grid.items():
            if isinstance(values, list):
                # Sample from list
                params[key] = self.rng.choice(values)
            else:
                # Assume it's a distribution with a sample method
                params[key] = values.rvs(random_state=self.rng)
        return params
    
    def fit(
        self,
        X: np.ndarray,
        bin_frequencies: np.ndarray,
        other_stats: np.ndarray,
        stat_names: List[str] = ['mean', 'std', 'q25', 'q75']
    ):
        """
        Perform randomized search.
        
        Parameters
        ----------
        X : np.ndarray
            Input parameters
        bin_frequencies : np.ndarray
            Bin frequency data
        other_stats : np.ndarray
            Summary statistics
        stat_names : list of str
            Names of statistics
        
        Returns
        -------
        self : RandomizedSearchCV
            Fitted tuner with results
        """
        if self.verbose > 0:
            print(f"\nRandomized Search: Testing {self.n_iter} parameter combinations")
            print(f"CV Folds: {self.cv}")
            print(f"Scoring: {self.scoring}")
            print("-"*60)
        
        results = []
        start_time = time.time()
        
        for i in range(self.n_iter):
            params = self._sample_params()
            
            if self.verbose > 0:
                print(f"[{i+1}/{self.n_iter}] Testing: {params}")
            
            cv_results = self._evaluate_params(
                params, X, bin_frequencies, other_stats, stat_names
            )
            score = self._get_score(cv_results)
            
            results.append({
                'params': params,
                'score': score,
                'cv_results': cv_results
            })
            
            if self.verbose > 1:
                print(f"  Score: {score:.4f}")
        
        elapsed = time.time() - start_time
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        self.results_ = results
        self.best_params_ = results[0]['params']
        self.best_score_ = results[0]['score']
        
        if self.verbose > 0:
            print(f"\nRandomized Search Complete!")
            print(f"Time elapsed: {elapsed:.2f}s")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best params: {self.best_params_}")
        
        return self


def tune_gp_kernel(
    X: np.ndarray,
    bin_frequencies: np.ndarray,
    other_stats: np.ndarray,
    search_type: str = 'grid',
    n_iter: int = 10,
    cv: int = 5,
    random_state: int = 42
) -> HyperparameterTuner:
    """
    Tune GP kernel hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    search_type : str, default='grid'
        'grid' for GridSearch or 'random' for RandomizedSearch
    n_iter : int, default=10
        Number of iterations for RandomizedSearch
    cv : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random state
    
    Returns
    -------
    tuner : HyperparameterTuner
        Fitted tuner with results
    """
    # Define parameter grid for GP
    param_grid = {
        'kernel': [
            Matern(length_scale=ls, nu=nu)
            for ls in [0.5, 1.0, 2.0]
            for nu in [0.5, 1.5, 2.5]
        ] + [
            RBF(length_scale=ls) + WhiteKernel(noise_level=noise)
            for ls in [0.5, 1.0, 2.0]
            for noise in [1e-6, 1e-5, 1e-4]
        ]
    }
    
    if search_type == 'grid':
        tuner = GridSearchCV(
            GPEmulatorBins,
            param_grid,
            cv=cv,
            scoring='bins_r2',
            random_state=random_state
        )
    else:
        tuner = RandomizedSearchCV(
            GPEmulatorBins,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='bins_r2',
            random_state=random_state
        )
    
    tuner.fit(X, bin_frequencies, other_stats)
    return tuner


def tune_nn_architecture(
    X: np.ndarray,
    bin_frequencies: np.ndarray,
    other_stats: np.ndarray,
    search_type: str = 'random',
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 42
) -> HyperparameterTuner:
    """
    Tune NN architecture hyperparameters.
    
    Parameters
    ----------
    X : np.ndarray
        Input parameters
    bin_frequencies : np.ndarray
        Bin frequency data
    other_stats : np.ndarray
        Summary statistics
    search_type : str, default='random'
        'grid' for GridSearch or 'random' for RandomizedSearch
    n_iter : int, default=20
        Number of iterations for RandomizedSearch
    cv : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random state
    
    Returns
    -------
    tuner : HyperparameterTuner
        Fitted tuner with results
    """
    # Define parameter grid for NN
    param_grid = {
        'n_parameters': [X.shape[1]],
        'n_bins': [bin_frequencies.shape[1]],
        'n_other_stats': [other_stats.shape[1]],
        'n_hidden_layers': [2, 3, 4],
        'n_nodes': [16, 32, 64, 128],
        'activation': ['relu', 'leaky_relu'],
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
        'l2_reg': [1e-5, 1e-4, 1e-3],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'patience': [20]
    }
    
    if search_type == 'grid':
        # For grid search, reduce the grid size to be more manageable
        param_grid_reduced = {
            'n_parameters': [X.shape[1]],
            'n_bins': [bin_frequencies.shape[1]],
            'n_other_stats': [other_stats.shape[1]],
            'n_hidden_layers': [2, 3],
            'n_nodes': [32, 64],
            'activation': ['relu'],
            'learning_rate': [1e-3, 5e-4],
            'l2_reg': [1e-4],
            'dropout_rate': [0.2, 0.3],
            'patience': [20]
        }
        
        tuner = GridSearchCV(
            NNEmulatorBins,
            param_grid_reduced,
            cv=cv,
            scoring='bins_r2',
            random_state=random_state
        )
    else:
        tuner = RandomizedSearchCV(
            NNEmulatorBins,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='bins_r2',
            random_state=random_state
        )
    
    tuner.fit(X, bin_frequencies, other_stats)
    return tuner
