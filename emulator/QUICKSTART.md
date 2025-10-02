# Quick Start Guide

This guide will get you up and running with the AMOC ML Tuning project in minutes.

## Installation (5 minutes)

```bash
# Create project directory
mkdir emulator
cd emulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Choose Your Model (1 minute)

Edit `config/config.py`:

```python
# Choose 'gp' or 'nn'
MODEL_TYPE = 'gp'  # Start with GP for small datasets with uncertainties
# MODEL_TYPE = 'nn'  # Use NN for faster training and larger datasets
```

## Prepare Your Data (10 minutes)

### Option 1: Use the data preparation module

```python
from src.data_preparation import load_amoc_data, process_ensemble_data, prepare_ml_data
from config.config import Config

# Load default run
amoc_default, time_default = load_amoc_data(Config.DEFAULT_RUN_FILE)

# Process ensemble runs
model_files = list(Config.ENSEMBLE_DIR.glob("*.nc"))
ensemble_stats = process_ensemble_data(model_files, n_bins=10)

# Prepare ML data
bin_frequencies, other_stats = prepare_ml_data(ensemble_stats)

# Load your input parameters (you need to provide this)
X = np.load('your_input_parameters.npy')  # shape: (n_samples, n_params)
```

### Option 2: Load pre-processed data

```python
import numpy as np

X = np.load('data/processed/input_parameters.npy')
bin_frequencies = np.load('data/processed/bin_frequencies.npy')
other_stats = np.load('data/processed/other_stats.npy')
```

## Train Your First Model (2 minutes)

### Using the training script

```python
from experiments.train_model import train_emulator
from config.config import Config

# Train model
model, test_data = train_emulator(X, bin_frequencies, other_stats, Config)

# Evaluate
from src.evaluation import evaluate_model, print_test_results

metrics = evaluate_model(
    model, 
    test_data['X_test'],
    test_data['Y_test'],
    test_data['bin_test'],
    Config.STAT_NAMES
)

print_test_results(metrics)
```

### Or use the command line

```bash
# Train with current config
python main.py --mode train

# Run cross-validation
python main.py --mode cv

# Compare different configurations
python main.py --mode compare
```

## Tune Hyperparameters (Optional but Recommended - 30-60 minutes)

Before training your final model, you can automatically find the best hyperparameters:

```bash
# Tune both models and find the best one
python tune_hyperparameters.py --model both --save

# Or tune just one model
python tune_hyperparameters.py --model gp --search grid
python tune_hyperparameters.py --model nn --search random --n-iter 20
```

The tuner will:
1. Try many parameter combinations
2. Use cross-validation to evaluate each
3. Report the best configuration
4. Save results to `tuning_results/`

Then update `config.py` with the best parameters!

For details, see [TUNING_GUIDE.md](TUNING_GUIDE.md).

## Compare GP vs NN (5 minutes)

```bash
# Quick comparison
python compare_models.py --mode simple

# Thorough comparison with cross-validation
python compare_models.py --mode cv

# Both
python compare_models.py --mode both
```

## Visualize Results (2 minutes)

```python
from src.visualization import plot_prediction_comparison

# For GP (with uncertainties)
if test_data['bin_std'] is not None:
    plot_prediction_comparison(
        test_data['bin_test'],
        test_data['bin_pred'],
        test_data['bin_std'],
        test_data['Y_test'],
        test_data['Y_pred'],
        n_bins=10,
        n_samples=3
    )

# For NN (without uncertainties)
else:
    from src.visualization import plot_prediction_comparison_no_std
    plot_prediction_comparison_no_std(
        test_data['bin_test'],
        test_data['bin_pred'],
        test_data['Y_test'],
        test_data['Y_pred'],
        n_bins=10,
        n_samples=3
    )
```

## Configuration Cheat Sheet

### For Small Datasets (<200 samples)

```python
# config.py
MODEL_TYPE = 'gp'
KERNEL = Matern(length_scale=1.0, nu=1.5)
CV_FOLDS = 5
```

Or if using NN:

```python
MODEL_TYPE = 'nn'
NN_N_HIDDEN_LAYERS = 2
NN_N_NODES = 32
NN_DROPOUT_RATE = 0.2
NN_L2_REG = 1e-4
```

### For Medium Datasets (200-1000 samples)

```python
MODEL_TYPE = 'nn'  # NN is faster and scales better
NN_N_HIDDEN_LAYERS = 3
NN_N_NODES = 64
NN_DROPOUT_RATE = 0.3
```

### For Large Datasets (>1000 samples)

```python
MODEL_TYPE = 'nn'  # Definitely use NN
NN_N_HIDDEN_LAYERS = 4
NN_N_NODES = 128
NN_LEARNING_RATE = 5e-4
NN_DROPOUT_RATE = 0.3
```

## Common Tasks

### Task 1: Try different GP kernels

```python
from sklearn.gaussian_process.kernels import Matern, RBF
from experiments.evaluate_model import compare_gp_kernels

comparison = compare_gp_kernels(X, bin_frequencies, other_stats)
```

### Task 2: Tune NN architecture

```python
from experiments.evaluate_model import compare_nn_architectures

comparison = compare_nn_architectures(X, bin_frequencies, other_stats)
```

### Task 3: Save and load models

```python
import pickle

# Save
with open('saved_models/my_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('saved_models/my_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Task 4: Make predictions on new data

```python
# Load trained model
model = ...  # Your trained model

# New parameter values
X_new = np.array([[param1, param2, ...]])  # shape: (n_new, n_params)

# Predict
if MODEL_TYPE == 'gp':
    bin_pred, stats_pred, bin_std, stats_std = model.predict(X_new, return_std=True)
    print(f"Predicted mean AMOC: {stats_pred[0, 0]:.2f} ± {stats_std[0, 0]:.2f} Sv")
else:  # NN
    bin_pred, stats_pred = model.predict(X_new)
    print(f"Predicted mean AMOC: {stats_pred[0, 0]:.2f} Sv")

print(f"Predicted std: {stats_pred[0, 1]:.2f} Sv")
print(f"Predicted Q25: {stats_pred[0, 2]:.2f} Sv")
print(f"Predicted Q75: {stats_pred[0, 3]:.2f} Sv")
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"

```bash
pip install tensorflow
# or for CPU only:
pip install tensorflow-cpu
```

### "GP training is too slow"

Switch to NN:

```python
MODEL_TYPE = 'nn'
```

Or use a simpler kernel:

```python
KERNEL = Matern(length_scale=1.0, nu=1.5)  # Faster than nu=2.5
```

### "NN is overfitting"

Increase regularization:

```python
NN_DROPOUT_RATE = 0.3  # Increase from 0.2
NN_L2_REG = 1e-3       # Increase from 1e-4
NN_PATIENCE = 10       # Stop earlier
```

Or reduce network size:

```python
NN_N_HIDDEN_LAYERS = 2  # Reduce from 3
NN_N_NODES = 32         # Reduce from 64
```

### "Poor predictions"

1. Check data quality and scaling
2. Try different model configurations
3. Increase training data if possible
4. Run cross-validation to diagnose issues

```python
from experiments.evaluate_model import run_cross_validation
results = run_cross_validation(X, bin_frequencies, other_stats)
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Experiment with hyperparameters** in `config/config.py`
3. **Add custom metrics** in `src/evaluation.py`
4. **Create custom visualizations** in `src/visualization.py`
5. **Compare models** using `compare_models.py`

## Project Structure Reference

```
amoc_ml_tuning/
├── src/                      # Core modules
│   ├── data_preparation.py   # Data loading and stats
│   ├── models.py            # GPEmulatorBins, NNEmulatorBins
│   ├── evaluation.py        # Metrics and cross-validation
│   ├── visualization.py     # Plotting functions
│   └── utils.py             # Helper functions
├── config/
│   └── config.py            # ⭐ Configure everything here
├── experiments/
│   ├── train_model.py       # Training script
│   └── evaluate_model.py    # Evaluation script
├── compare_models.py         # GP vs NN comparison
├── main.py                   # Main entry point
└── requirements.txt          # Dependencies
```

## Quick Reference: Model APIs

### GPEmulatorBins

```python
from src.models import GPEmulatorBins

model = GPEmulatorBins(kernel=your_kernel)
model.fit(X_train, Y_train, bin_train)
bin_pred, Y_pred, bin_std, Y_std = model.predict(X_test, return_std=True)
scores = model.score(X_test, Y_test, bin_test)
```

### NNEmulatorBins

```python
from src.models import NNEmulatorBins

model = NNEmulatorBins(
    n_parameters=X.shape[1],
    n_bins=10,
    n_other_stats=4,
    n_hidden_layers=2,
    n_nodes=32
)
model.fit(X_train, Y_train, bin_train, epochs=100, batch_size=8)
bin_pred, Y_pred = model.predict(X_test)
scores = model.score(X_test, Y_test, bin_test)
```

## Getting Help

- Check the main **README.md** for detailed documentation
- Look at **example code** in the scripts
- Review the **docstrings** in each module
- Test with **small datasets** first before scaling up

Good luck with your AMOC emulation project! 🌊
