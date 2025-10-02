"""
AMOC ML Tuning Package

Machine learning tools for AMOC emulation and model tuning.
Supports both Gaussian Process and Neural Network emulators.
"""

from .data_preparation import (
    compute_simple_summary_stats,
    load_amoc_data,
    process_ensemble_data,
    prepare_ml_data,
    get_amoc_bins
)

from .models import GPEmulatorBins

# Try to import NN emulator (requires TensorFlow)
try:
    from .models import NNEmulatorBins
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False
    NNEmulatorBins = None

from .evaluation import (
    crossvalidation_bins,
    evaluate_model,
    print_cv_results,
    print_test_results
)

from .visualization import (
    plot_amoc_ensemble_simple,
    plot_prediction_comparison,
    plot_prediction_comparison_no_std
)

from .tuning import (
    GridSearchCV,
    RandomizedSearchCV,
    tune_gp_kernel,
    tune_nn_architecture
)

__version__ = '0.1.0'

__all__ = [
    'compute_simple_summary_stats',
    'load_amoc_data',
    'process_ensemble_data',
    'prepare_ml_data',
    'get_amoc_bins',
    'GPEmulatorBins',
    'NNEmulatorBins',
    'NN_AVAILABLE',
    'crossvalidation_bins',
    'evaluate_model',
    'print_cv_results',
    'print_test_results',
    'plot_amoc_ensemble_simple',
    'plot_prediction_comparison',
    'plot_prediction_comparison_no_std',
    'GridSearchCV',
    'RandomizedSearchCV',
    'tune_gp_kernel',
    'tune_nn_architecture',
]
"""
AMOC ML Tuning Package

Machine learning tools for AMOC emulation and model tuning.
"""

from .data_preparation import (
    compute_simple_summary_stats,
    load_amoc_data,
    process_ensemble_data,
    prepare_ml_data,
    get_amoc_bins
)

from .models import GPEmulatorBins

from .evaluation import (
    crossvalidation_bins,
    evaluate_model,
    print_cv_results,
    print_test_results
)

from .visualization import (
    plot_amoc_ensemble_simple,
    plot_prediction_comparison
)

__version__ = '0.1.0'

__all__ = [
    'compute_simple_summary_stats',
    'load_amoc_data',
    'process_ensemble_data',
    'prepare_ml_data',
    'get_amoc_bins',
    'GPEmulatorBins',
    'crossvalidation_bins',
    'evaluate_model',
    'print_cv_results',
    'print_test_results',
    'plot_amoc_ensemble_simple',
    'plot_prediction_comparison',
]
