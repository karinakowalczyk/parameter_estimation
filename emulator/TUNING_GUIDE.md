# Hyperparameter Tuning Guide

This guide explains how to properly tune hyperparameters for both GP and NN emulators using GridSearch and RandomizedSearch.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Hyperparameters](#understanding-hyperparameters)
3. [GridSearch vs RandomizedSearch](#gridsearch-vs-randomizedsearch)
4. [Tuning Strategies](#tuning-strategies)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

## Quick Start

### Command Line Interface

```bash
# Tune both models and save results
python tune_hyperparameters.py --model both --save

# Tune only GP with grid search
python tune_hyperparameters.py --model gp --search grid

# Tune only NN with 30 random configurations
python tune_hyperparameters.py --model nn --search random --n-iter 30
```

### Python API

```python
from src.tuning import GridSearchCV, tune_gp_kernel, tune_nn_architecture

# Quick GP tuning
gp_tuner = tune_gp_kernel(X, bin_frequencies, other_stats, search_type='grid')
print(f"Best GP kernel: {gp_tuner.best_params_}")

# Quick NN tuning  
nn_tuner = tune_nn_architecture(X, bin_frequencies, other_stats, n_iter=20)
print(f"Best NN architecture: {nn_tuner.best_params_}")
```

## Understanding Hyperparameters

### Gaussian Process Hyperparameters

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|---------------|
| **kernel type** | RBF, Matern, etc. | Smoothness assumptions | - |
| **length_scale** | Controls correlation distance | Too small: overfitting; Too large: underfitting | 0.1 - 10.0 |
| **nu (Matern)** | Smoothness parameter | 0.5=rough, 1.5=moderate, 2.5=smooth | 0.5, 1.5, 2.5 |
| **noise_level** | Observation noise | Too high: underfits data | 1e-6 - 1e-3 |

**Key insight**: GP kernels encode assumptions about the function you're approximating. For AMOC data, Matern ν=1.5 often works well.

### Neural Network Hyperparameters

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|---------------|
| **n_hidden_layers** | Network depth | More layers = more capacity | 2-4 (small data) |
| **n_nodes** | Width of each layer | More nodes = more capacity | 16-128 |
| **learning_rate** | Step size for optimization | Too high: unstable; Too low: slow | 1e-4 - 5e-3 |
| **dropout_rate** | Regularization strength | Higher = more regularization | 0.1-0.4 |
| **l2_reg** | Weight decay | Prevents large weights | 1e-5 - 1e-3 |
| **activation** | Nonlinearity | ReLU is standard | relu, leaky_relu |
| **batch_size** | Samples per update | Smaller = more updates | 4-32 (small data) |

**Key insight**: For small datasets (<200 samples), use smaller networks (2-3 layers, 32-64 nodes) with strong regularization (dropout 0.2-0.3, L2 1e-4).

## GridSearch vs RandomizedSearch

### GridSearch

**Pros:**
- Exhaustive: tries all combinations
- Reproducible
- Good for small search spaces

**Cons:**
- Exponentially slow with more parameters
- Inefficient for large search spaces

**When to use:**
- GP tuning (few parameters)
- Small NN searches (2-3 parameters)
- When computational budget allows
- When you need exhaustive testing

**Example timing:**
- GP: 10-20 kernel combinations = 5-15 minutes
- NN: 2×2×2 = 8 combinations = 10-30 minutes

### RandomizedSearch

**Pros:**
- Samples randomly from distributions
- Can cover large search spaces efficiently
- Budget-controlled (set n_iter)
- Often finds good solutions faster

**Cons:**
- Not exhaustive
- May miss optimal configuration
- Slightly less reproducible

**When to use:**
- NN tuning (many parameters)
- Large search spaces
- Limited time/computation
- Exploratory tuning

**Example timing:**
- GP: 10 random samples = 3-8 minutes
- NN: 20 random samples = 20-60 minutes

### Recommendation

- **GP**: Use GridSearch (small search space)
- **NN**: Use RandomizedSearch with n_iter=20-50

## Tuning Strategies

### Strategy 1: Coarse-to-Fine (Recommended)

1. **Coarse search**: Wide ranges, few samples
2. **Analyze results**: Identify promising regions
3. **Fine search**: Narrow ranges around best results

```python
# Step 1: Coarse search
param_grid_coarse = {
    'n_hidden_layers': [2, 3, 4],
    'n_nodes': [32, 64, 128],
    'learning_rate': [1e-3, 1e-2]
}
tuner_coarse = RandomizedSearchCV(NNEmulatorBins, param_grid_coarse, n_iter=10)
tuner_coarse.fit(X, bin_frequencies, other_stats)

# Step 2: Fine search around best parameters
best = tuner_coarse.best_params_
param_grid_fine = {
    'n_hidden_layers': [best['n_hidden_layers']],
    'n_nodes': [best['n_nodes'] // 2, best['n_nodes'], best['n_nodes'] * 2],
    'learning_rate': [best['learning_rate'] / 2, best['learning_rate'], best['learning_rate'] * 2]
}
tuner_fine = GridSearchCV(NNEmulatorBins, param_grid_fine)
tuner_fine.fit(X, bin_frequencies, other_stats)
```

### Strategy 2: Dataset-Size Based

**Small datasets (<100 samples):**
```python
# Prioritize regularization
param_grid = {
    'n_hidden_layers': [2],        # Shallow
    'n_nodes': [16, 32],           # Small
    'dropout_rate': [0.3, 0.4],    # High dropout
    'l2_reg': [1e-4, 1e-3]        # Strong L2
}
```

**Medium datasets (100-500 samples):**
```python
param_grid = {
    'n_hidden_layers': [2, 3],
    'n_nodes': [32, 64],
    'dropout_rate': [0.2, 0.3],
    'l2_reg': [1e-4, 1e-5]
}
```

**Large datasets (>500 samples):**
```python
param_grid = {
    'n_hidden_layers': [3, 4],
    'n_nodes': [64, 128, 256],
    'dropout_rate': [0.1, 0.2],
    'l2_reg': [1e-5, 1e-6]
}
```

### Strategy 3: Sequential Tuning

Tune parameters in groups:

```python
# Step 1: Architecture
param_grid_arch = {
    'n_hidden_layers': [2, 3, 4],
    'n_nodes': [32, 64, 128],
    'learning_rate': [1e-3],  # Fixed
    'dropout_rate': [0.2]      # Fixed
}

# Step 2: Optimization (using best architecture)
param_grid_opt = {
    'n_hidden_layers': [best_layers],  # Fixed
    'n_nodes': [best_nodes],           # Fixed
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4]
}

# Step 3: Regularization (using best architecture + optimizer)
param_grid_reg = {
    'n_hidden_layers': [best_layers],
    'n_nodes': [best_nodes],
    'learning_rate': [best_lr],
    'dropout_rate': [best_dropout],
    'l2_reg': [1e-6, 1e-5, 1e-4, 1e-3]
}
```

## Examples

### Example 1: Basic GP Tuning

```python
from src.tuning import GridSearchCV
from src.models import GPEmulatorBins
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

# Define search space
param_grid = {
    'kernel': [
        Matern(length_scale=0.5, nu=1.5),
        Matern(length_scale=1.0, nu=1.5),
        Matern(length_scale=2.0, nu=1.5),
        Matern(length_scale=1.0, nu=0.5),
        Matern(length_scale=1.0, nu=2.5),
        RBF(length_scale=1.0) + WhiteKernel(1e-6),
    ]
}

# Run grid search
tuner = GridSearchCV(
    GPEmulatorBins,
    param_grid,
    cv=5,
    scoring='bins_r2',
    verbose=1
)
tuner.fit(X, bin_frequencies, other_stats)

# View results
tuner.print_results(top_n=5)

# Use best model
best_kernel = tuner.best_params_['kernel']
final_model = GPEmulatorBins(kernel=best_kernel)
final_model.fit(X_train, Y_train, bin_train)
```

### Example 2: NN Random Search

```python
from src.tuning import RandomizedSearchCV
from src.models import NNEmulatorBins

# Define search space
param_distributions = {
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

# Run random search
tuner = RandomizedSearchCV(
    NNEmulatorBins,
    param_distributions,
    n_iter=30,
    cv=5,
    scoring='bins_r2',
    random_state=42,
    verbose=1
)
tuner.fit(X, bin_frequencies, other_stats)

# View results
tuner.print_results(top_n=10)

# Update config with best parameters
print("\nAdd these to config.py:")
for key, value in tuner.best_params_.items():
    if key.startswith('NN_'):
        continue
    param_name = 'NN_' + key.upper()
    print(f"{param_name} = {value}")
```

### Example 3: Compare Multiple Scoring Metrics

```python
from src.tuning import GridSearchCV

# Try different scoring metrics
metrics = ['bins_r2', 'bins_rmse', 'mean_rmse']
results_by_metric = {}

for metric in metrics:
    print(f"\nTuning with metric: {metric}")
    tuner = GridSearchCV(
        GPEmulatorBins,
        param_grid,
        cv=5,
        scoring=metric,
        verbose=0
    )
    tuner.fit(X, bin_frequencies, other_stats)
    results_by_metric[metric] = tuner.best_params_

# Compare
print("\nBest parameters by metric:")
for metric, params in results_by_metric.items():
    print(f"{metric}: {params}")
```

### Example 4: Save and Load Tuning Results

```python
import json
from pathlib import Path

# Run tuning
tuner = RandomizedSearchCV(NNEmulatorBins, param_grid, n_iter=20)
tuner.fit(X, bin_frequencies, other_stats)

# Save results
results_dir = Path('tuning_results')
results_dir.mkdir(exist_ok=True)

tuning_data = {
    'best_params': tuner.best_params_,
    'best_score': float(tuner.best_score_),
    'top_10_configs': [
        {'params': str(r['params']), 'score': float(r['score'])}
        for r in tuner.results_[:10]
    ]
}

with open(results_dir / 'nn_tuning_results.json', 'w') as f:
    json.dump(tuning_data, f, indent=2)

print(f"Results saved to {results_dir / 'nn_tuning_results.json'}")
```

## Best Practices

### 1. Start Simple

Begin with default configurations before tuning:
- GP: Matern(1.0, nu=1.5)
- NN: 2 layers, 32 nodes, lr=1e-3

Only tune if performance is unsatisfactory.

### 2. Use Cross-Validation

Always use CV (k=5 is standard) to avoid overfitting to a single train/test split.

### 3. Monitor Overfitting

Look for large gaps between train and validation performance:

```python
# Check if model is overfitting
if train_score - val_score > 0.1:
    print("Model is overfitting! Increase regularization:")
    print("  - Increase dropout_rate")
    print("  - Increase l2_reg")
    print("  - Reduce network size")
```

### 4. Budget Your Time

**Limited time (<30 min):**
- Use pre-defined configs from `config.py`
- Try 3-5 configurations manually

**Moderate time (1-2 hours):**
- RandomSearch with n_iter=20 for NN
- GridSearch for GP

**Extensive time (>2 hours):**
- Coarse-to-fine strategy
- Try both GP and NN
- Multiple scoring metrics

### 5. Interpret Results

Don't just pick the best score:

```python
tuner.print_results(top_n=10)

# Look for:
# 1. Consistent patterns (e.g., always prefer 2 layers)
# 2. Small score differences (top 5 might be equivalent)
# 3. Simpler models with similar performance
# 4. Suspicious outliers
```

### 6. Test on Hold-out Set

After tuning, evaluate on completely fresh data:

```python
# Split data BEFORE tuning
X_dev, X_test, ... = train_test_split(X, ..., test_size=0.2)

# Tune on development set only
tuner.fit(X_dev, bin_dev, stats_dev)

# Final evaluation on test set
final_model = NNEmulatorBins(**tuner.best_params_)
final_model.fit(X_dev, Y_dev, bin_dev)
test_score = final_model.score(X_test, Y_test, bin_test)
```

### 7. Document Your Findings

Keep a log of tuning experiments:

```python
import datetime

log_entry = {
    'date': str(datetime.datetime.now()),
    'dataset_size': len(X),
    'search_type': 'random',
    'n_iter': 20,
    'best_score': tuner.best_score_,
    'best_params': tuner.best_params_,
    'notes': 'Tried larger networks, didn't help'
}

# Append to experiment log
with open('tuning_log.jsonl', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
```

## Common Issues

### Issue: Tuning takes too long

**Solutions:**
- Reduce CV folds (5 → 3)
- Use RandomSearch instead of Grid
- Reduce n_iter
- Parallelize (not implemented yet)
- Use a subset of data for initial tuning

### Issue: All configurations perform similarly

This means:
- Dataset might be too easy/small
- Current features might be very informative
- **Action**: Choose simplest model (fewer parameters)

### Issue: Large variance across CV folds

This means:
- Dataset might have high variability
- Train/test splits are very different
- **Action**: Increase CV folds, get more data, or check for data issues

### Issue: Best model overfits

**Solutions:**
- Increase regularization (dropout, L2)
- Reduce model size
- Get more training data
- Check for data leakage

## Computational Budget Guidelines

Based on typical AMOC dataset sizes:

| Dataset Size | Model | Search Type | n_iter | Time | Recommendation |
|--------------|-------|-------------|--------|------|----------------|
| <50 samples | GP | Grid | - | 10-20 min | Use GP only |
| 50-100 | GP | Grid | - | 10-20 min | GP primary |
| 50-100 | NN | Random | 10-15 | 15-30 min | NN secondary |
| 100-200 | Both | Grid/Random | 20 | 30-60 min | Compare both |
| 200-500 | NN | Random | 30-50 | 60-120 min | NN primary |
| >500 | NN | Random | 50-100 | 120-300 min | NN only |

## Next Steps

After tuning:

1. **Update config.py** with best parameters
2. **Train final model** on full dataset
3. **Evaluate thoroughly** with cross-validation
4. **Document** which configurations worked
5. **Save model** for future use

```bash
# After tuning, update config and train
python main.py --mode train

# Evaluate with cross-validation
python main.py --mode cv

# Save model
# (automatically done if SAVE_MODELS=True in config)
```

Good luck with your hyperparameter tuning! 🎯
