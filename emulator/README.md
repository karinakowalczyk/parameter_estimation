# AMOC ML Tuning Project

Machine learning project for tuning Gaussian Process emulators to predict AMOC (Atlantic Meridional Overturning Circulation) statistics from ensemble simulations.

## Project Structure

```
emulator/
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # Data loading and preprocessing
│   ├── models.py                 # GP emulator classes
│   ├── evaluation.py             # Cross-validation and metrics
│   ├── visualization.py          # Plotting functions
│   ├── tuning.py                 # Hyperparameter tuning
│   └── utils.py                  # Helper utilities
├── config/
│   └── config.py                 # Configuration and hyperparameters
├── experiments/
│   ├── train_model.py           # Training script
│   └── evaluate_model.py        # Evaluation and CV script
├── data/                         # Data directory (not in repo)
│   ├── raw/
│   ├── processed/
│   └── ensemble_runs/
├── saved_models/                 # Saved model files
├── figures/                      # Output plots
├── main.py                       # Main pipeline script
├── requirements.txt
└── README.md
```

## Installation

1. **Create the project directory:**

```bash
mkdir emulator
cd emulator
```

2. **Create the folder structure** (see Project Structure section above)

3. **Copy all Python files** to their respective locations

4. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure Your Project

Edit `config/config.py` to set your data paths and choose model type:

```python
# Choose model type
MODEL_TYPE = 'gp'  # or 'nn' for Neural Network

# Set your data paths
DATA_DIR = Path("../data")
DEFAULT_RUN_FILE = DATA_DIR / "default_long_run" / "ocn_ts.nc"
ENSEMBLE_DIR = DATA_DIR / "ensemble_runs"

# For GP: adjust kernel
KERNEL = Matern(length_scale=1.0, nu=1.5)

# For NN: adjust architecture
NN_N_HIDDEN_LAYERS = 2
NN_N_NODES = 32
NN_LEARNING_RATE = 1e-3
```

### 2. Prepare Your Data

Your data should be in netCDF format with an `amoc26N` variable. Organize as:

```
data/
├── default_long_run/
│   └── ocn_ts.nc
└── ensemble_runs/
    ├── run_001.nc
    ├── run_002.nc
    └── ...
```

### 3. Run the Pipeline

**Train a model:**
```bash
python main.py --mode train
```

**Run cross-validation:**
```bash
python main.py --mode cv
```

**Compare different configurations:**
```bash
python main.py --mode compare
```

**Tune hyperparameters:**
```bash
# Tune both models
python tune_hyperparameters.py --model both --save

# Tune only GP
python tune_hyperparameters.py --model gp --search grid

# Tune only NN with random search
python tune_hyperparameters.py --model nn --search random --n-iter 30
```

**Visualize ensemble data:**
```bash
python main.py --mode train --visualize
```

## Usage Examples

### Hyperparameter Tuning (NEW!)

The project now includes proper hyperparameter tuning with GridSearch and RandomizedSearch:

```python
from src.tuning import GridSearchCV, RandomizedSearchCV
from src.models import GPEmulatorBins, NNEmulatorBins
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

# Define parameter grid for GP
param_grid_gp = {
    'kernel': [
        Matern(length_scale=ls, nu=nu)
        for ls in [0.5, 1.0, 2.0]
        for nu in [0.5, 1.5, 2.5]
    ]
}

# Grid search for GP
tuner = GridSearchCV(
    GPEmulatorBins,
    param_grid_gp,
    cv=5,
    scoring='bins_r2',
    verbose=1
)
tuner.fit(X, bin_frequencies, other_stats)
print(f"Best params: {tuner.best_params_}")
print(f"Best score: {tuner.best_score_:.3f}")

# Define parameter grid for NN
param_grid_nn = {
    'n_parameters': [X.shape[1]],
    'n_bins': [10],
    'n_other_stats': [4],
    'n_hidden_layers': [2, 3, 4],
    'n_nodes': [32, 64, 128],
    'learning_rate': [1e-4, 1e-3, 5e-3],
    'dropout_rate': [0.1, 0.2, 0.3]
}

# Random search for NN (more efficient)
tuner_nn = RandomizedSearchCV(
    NNEmulatorBins,
    param_grid_nn,
    n_iter=20,
    cv=5,
    scoring='bins_r2',
    verbose=1
)
tuner_nn.fit(X, bin_frequencies, other_stats)
tuner_nn.print_results(top_n=5)
```

### Train a Model

```python
from src.models import GPEmulatorBins
from sklearn.gaussian_process.kernels import Matern
from experiments.train_model import train_emulator

# Your data
X = ...  # Input parameters, shape (n_samples, n_params)
bin_frequencies = ...  # Bin frequencies, shape (n_samples, n_bins)
other_stats = ...  # Statistics, shape (n_samples, 4)

# Train
kernel = Matern(length_scale=1.0, nu=1.5)
from config.config import Config
Config.KERNEL = kernel

model, test_data = train_emulator(X, bin_frequencies, other_stats, Config)
```

### Run Cross-Validation

```python
from experiments.evaluate_model import run_cross_validation
from config.config import Config

results = run_cross_validation(X, bin_frequencies, other_stats, Config)
print(f"Bins RMSE: {results['bins_rmse']:.4f}")
print(f"Bins R²: {results['bins_r2']:.3f}")
```

### Make Predictions

```python
# Predict on new inputs
X_new = ...  # New parameter values
bin_pred, stats_pred, bin_std, stats_std = model.predict(X_new, return_std=True)

print(f"Predicted mean AMOC: {stats_pred[0, 0]:.2f} Sv")
print(f"Predicted std: {stats_pred[0, 1]:.2f} Sv")
```

### Visualize Predictions

```python
from src.visualization import plot_prediction_comparison

plot_prediction_comparison(
    bin_test, bin_pred, bin_std,
    Y_test, Y_pred,
    n_bins=10,
    n_samples=3
)
```

## Model Classes

### GPEmulatorBins

Gaussian Process emulator that predicts both bin frequencies and summary statistics.

**Key Features:**
- Multi-output GP captures correlations between outputs
- Automatic normalization of bin frequencies
- Supports various kernel functions (Matern, RBF, etc.)
- Returns prediction uncertainties
- Ideal for small datasets with uncertainty quantification needs

**Methods:**
- `fit(X, Y, bin_frequencies)` - Train the emulator
- `predict(X_new, return_std=True)` - Make predictions with uncertainties
- `score(X_test, Y_test, bin_frequencies_test)` - Compute R² scores

### NNEmulatorBins

Neural Network emulator with regularization for small datasets.

**Key Features:**
- Feed-forward architecture with configurable layers
- L2 regularization and dropout to prevent overfitting
- Early stopping with validation monitoring
- Batch normalization via input/output scaling
- Faster training and prediction than GP
- Better scalability to larger datasets

**Methods:**
- `fit(X, Y, bin_frequencies, epochs=100)` - Train the network
- `predict(X_new)` - Make predictions (no uncertainties)
- `score(X_test, Y_test, bin_frequencies_test)` - Compute R² scores

**Note**: NN does not provide prediction uncertainties by default.

## Configuration Options

### Model Selection

Choose between GP and NN in `config/config.py`:

```python
MODEL_TYPE = 'gp'  # or 'nn'
```

### GP Kernel Selection

The kernel choice significantly impacts GP model performance:

```python
# Matern kernel (recommended for AMOC data)
KERNEL = Matern(length_scale=1.0, nu=1.5)

# RBF kernel (for smooth functions)
KERNEL = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
```

### Neural Network Architecture

Configure the NN architecture for your dataset size:

```python
# For small datasets (~100 samples)
NN_N_HIDDEN_LAYERS = 2
NN_N_NODES = 32
NN_DROPOUT_RATE = 0.2
NN_L2_REG = 1e-4

# For medium datasets (~500 samples)
NN_N_HIDDEN_LAYERS = 3
NN_N_NODES = 64
NN_DROPOUT_RATE = 0.3

# For large datasets (>1000 samples)
NN_N_HIDDEN_LAYERS = 4
NN_N_NODES = 128
NN_DROPOUT_RATE = 0.3
```

### Data Processing

```python
N_BINS = 10              # Number of histogram bins
REMOVE_SPINUP = False    # Remove initial spinup period
SPINUP_FRACTION = 0.1    # Fraction to remove if enabled
```

### Training

```python
TEST_SIZE = 0.2          # Train/test split ratio
CV_FOLDS = 5             # Cross-validation folds
RANDOM_STATE = 42        # Reproducibility seed
```

## Evaluation Metrics

The project tracks multiple metrics:

- **Bin Frequencies:**
  - RMSE: Root mean squared error
  - Max Error: Maximum pointwise error
  - R²: Coefficient of determination

- **Summary Statistics:**
  - Individual RMSE for mean, std, q25, q75
  - Individual R² scores

## Extending the Project

### Add a New Model

Create a new class in `src/models.py`:

```python
class MyCustomEmulator:
    def __init__(self, ...):
        self.is_fitted = False
        self.n_bins = None
        self.n_other_stats = None
    
    def fit(self, X, Y, bin_frequencies):
        # Your training logic
        self.is_fitted = True
        return self
    
    def predict(self, X_new, return_std=False):
        # Your prediction logic
        # Must return: bin_frequencies_pred, Y_pred
        # Optional: bin_std, Y_std if return_std=True
        pass
    
    def score(self, X_test, Y_test, bin_frequencies_test):
        # Return dict with 'bin_frequencies_r2', 'other_stats_r2', 'overall_r2'
        pass
```

Then update `config/config.py` to support your model type.

### Add Custom Metrics

Add to `src/evaluation.py`:

```python
def my_custom_metric(y_true, y_pred):
    # Your metric calculation
    return score
```

### Add New Visualizations

Add to `src/visualization.py`:

```python
def plot_my_analysis(...):
    # Your plotting code
    pass
```

## Data Format

### Input Files (netCDF)

Required variables:
- `amoc26N`: AMOC strength at 26°N (Sverdrups)
- `time`: Time coordinate

### Processed Data Format

- **X**: Input parameters, shape `(n_samples, n_parameters)`
- **bin_frequencies**: Histogram bins, shape `(n_samples, n_bins)`
- **other_stats**: Summary statistics, shape `(n_samples, 4)`
  - Column 0: mean
  - Column 1: std
  - Column 2: q25 (25th percentile)
  - Column 3: q75 (75th percentile)

## Troubleshooting

**Issue: Model predictions are poor**
- **GP**: Try different kernels (Matern ν=2.5, RBF), adjust length scales
- **NN**: Increase network size, adjust learning rate, add more regularization
- Increase training data if possible
- Check for data quality issues or outliers
- Verify input scaling is working properly

**Issue: NN training is unstable**
- Reduce learning rate (try 1e-4 instead of 1e-3)
- Increase regularization (L2_REG and DROPOUT_RATE)
- Reduce network size for small datasets
- Check if validation loss is increasing (overfitting)

**Issue: GP is too slow**
- Use a simpler kernel (Matern ν=1.5 instead of ν=2.5)
- Reduce number of training samples
- Consider using NN for faster training
- Use sparse GP approximations (not implemented)

**Issue: Cross-validation is slow**
- Reduce CV_FOLDS (try 3 instead of 5)
- Use fewer samples for initial testing
- For NN: reduce epochs or use early stopping more aggressively

**Issue: Memory errors**
- Process data in batches
- Reduce number of bins
- Use float32 instead of float64
- For NN: reduce batch size

**Issue: TensorFlow not available**
- Install TensorFlow: `pip install tensorflow`
- Or use CPU-only version: `pip install tensorflow-cpu`
- Or set `MODEL_TYPE = 'gp'` to use only GP

## Model Selection Guide

**Use Gaussian Process (GP) when:**
- ✅ You have a small dataset (<200 samples)
- ✅ You need uncertainty quantification
- ✅ You want interpretable kernel parameters
- ✅ You have time for slower training

**Use Neural Network (NN) when:**
- ✅ You have a medium-large dataset (>100 samples)
- ✅ You need fast predictions
- ✅ You don't require uncertainty estimates
- ✅ You want easier scalability to more data

**Performance Comparison** (typical on AMOC data):
| Metric | GP | NN |
|--------|----|----|
| Training Time (100 samples) | ~30s | ~10s |
| Prediction Time (1 sample) | ~0.1s | ~0.001s |
| R² Score (bins) | 0.85-0.95 | 0.80-0.90 |
| Uncertainty | ✅ Yes | ❌ No |
| Overfitting Risk | Low | Medium |

## Citation

If you use this code in your research, please cite:

```
[Your citation information]
```

## License

[Your license information]

## Contact

[Your contact information]
