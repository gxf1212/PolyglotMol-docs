# Grid Search

The grid search module provides tools for automated model selection and hyperparameter optimization. It supports both regression and classification tasks, with integrated data splitting, evaluation, and result tracking.

```{toctree}
:maxdepth: 1
:hidden:
```

## Overview

PolyglotMol's grid search functionality automatically:
- Tests multiple machine learning models
- Optimizes hyperparameters for each model
- Evaluates different molecular representations
- Tracks and saves all results
- Identifies the best model/representation combination

## Basic Usage with MolecularDataset

The simplest way to use grid search is with a `MolecularDataset`:

```python
from polyglotmol.data import MolecularDataset
from polyglotmol.models import RegressionGridSearch
import polyglotmol as pm

# Load data
dataset = MolecularDataset.from_csv(
    filepath="activity_data.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Add molecular features
dataset.add_features(["morgan_fp_r2_1024", "maccs_keys", "rdkit_descriptors"])

# Initialize grid search
gs = RegressionGridSearch(
    output_dir="./grid_search_results",
    random_state=42,
    n_jobs=-1,  # Use all CPU cores
    cv=5,       # 5-fold cross-validation
    comparison_metric='r2'  # Compare models by R² score
)

# Run grid search on all features in the dataset
results = gs.run_grid_search(
    dataset=dataset,
    target_column="activity",
    preprocessing='standard',
    test_size=0.2
)

# Get the best model
best_model, best_result = gs.get_best_model()
print(f"Best model: {best_result.model_name} with {best_result.representation_name}")
print(f"R² score: {best_result.metrics['r2']:.4f}")
```

## Selecting Specific Features

You can choose which features to include in the grid search:

```python
# Use only specific features
results = gs.run_grid_search(
    dataset=dataset,
    target_column="activity",
    representations_to_use=['morgan_fp_r2_1024', 'maccs_keys'],  # Only these
    preprocessing='standard',
    test_size=0.2
)
```

## Filtering Data Before Grid Search

Filter your dataset based on error values or other criteria:

```python
# Load dataset with error columns
dataset = MolecularDataset.from_df(
    df=endo_df,
    input_column='SMILES',
    id_column='Identifier',
    label_columns=['Bx'],
    label_error_suffix='-stderr'  # Automatically detects 'Bx-stderr'
)

# Filter by error threshold
dataset = dataset.filter_by_error('Bx', max_error=10.0)

# Add features
featurizers = pm.create_featurizer_ensemble(
    "fingerprints/datamol", "fingerprints/cdk",
    max_featurizers=50
)
dataset.add_features(featurizers, n_workers=16)

# Run grid search on filtered data
gs = RegressionGridSearch(
    output_dir="./filtered_results",
    comparison_metric='mae'  # Use MAE for comparison
)

results = gs.run_grid_search(
    dataset=dataset,
    target_column="Bx",
    preprocessing='standard'
)
```

## Classification Tasks

For classification problems, use `ClassificationGridSearch`:

```python
from polyglotmol.models import ClassificationGridSearch

# Create classification dataset
dataset = MolecularDataset.from_csv(
    filepath="classification_data.csv",
    input_column="SMILES",
    label_columns=["activity_class"]  # Binary or multi-class labels
)

# Add features
dataset.add_features(["ecfp4_1024", "maccs_keys"])

# Initialize classification grid search
gs = ClassificationGridSearch(
    output_dir="./classification_results",
    random_state=42,
    n_jobs=-1,
    cv=5,
    comparison_metric='f1'  # Compare by F1 score
)

# Run grid search
results = gs.run_grid_search(
    dataset=dataset,
    target_column="activity_class",
    preprocessing='standard'
)

# Get best model by different metrics
best_f1 = gs.get_best_result(metric="f1", higher_is_better=True)
best_accuracy = gs.get_best_result(metric="accuracy", higher_is_better=True)

print(f"Best F1 score: {best_f1.metrics['f1']:.4f} ({best_f1.model_name})")
print(f"Best accuracy: {best_accuracy.metrics['accuracy']:.4f} ({best_accuracy.model_name})")
```

## Custom Model Selection

Specify which models to include or exclude:

```python
# Use only specific models
results = gs.run_grid_search(
    dataset=dataset,
    target_column="activity",
    models={
        'Random Forest': RandomForestRegressor(),
        'XGBoost': XGBRegressor(),
        'Neural Network': MLPRegressor()
    }
)

# Or use convenience function with model names
from polyglotmol.models import run_regression_grid_search

gs = run_regression_grid_search(
    dataset=dataset,
    feature_column="morgan_fp_r2_1024",
    target_column="activity",
    models=["Random Forest", "Gradient Boosting", "Ridge"],
    exclude_models=["Linear Regression"]  # Exclude specific models
)
```

## Tracking and Analyzing Results

Grid search provides comprehensive result tracking:

```python
# Get top N results
top_5_results = gs.get_best_results(metric='r2', higher_is_better=True, top_n=5)

for i, result in enumerate(top_5_results, 1):
    print(f"{i}. {result.model_name} + {result.representation_name}")
    print(f"   R² = {result.metrics['r2']:.4f}, RMSE = {result.metrics['rmse']:.4f}")
    print(f"   Training time: {result.training_time:.2f}s")

# Save results summary to CSV
summary_path = gs.save_results_summary()
print(f"Results saved to: {summary_path}")

# Access individual results
for result in gs.results:
    if result.model_name == "Random Forest":
        print(f"RF with {result.representation_name}: {result.metrics}")
```

## Examining Model Details

Each `ModelResult` contains detailed information:

```python
# Get the best result
best_model, best_result = gs.get_best_model()

# Access comprehensive information
print(f"Model: {best_result.model_name}")
print(f"Representation: {best_result.representation_name}")
print(f"Parameters: {best_result.params}")
print(f"All metrics: {best_result.metrics}")
print(f"Training time: {best_result.training_time:.2f}s")

# Feature importance (if available)
if best_result.feature_importance:
    top_features = sorted(
        best_result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    print("Top 10 important features:")
    for feat, importance in top_features:
        print(f"  {feat}: {importance:.4f}")

# Save and load results
best_result.save("./best_model_result.json")

# Load later
from polyglotmol.models.grid_search import ModelResult
loaded_result = ModelResult.load("./best_model_result.json")
```

## Using Custom Data

If you have pre-computed features, you can use them directly:

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Prepare your own features and labels
X = np.random.rand(1000, 100)  # 1000 samples, 100 features
y = np.random.rand(1000)        # Target values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize grid search
gs = RegressionGridSearch(output_dir="./custom_results")

# Set the data attributes
gs._X_train = X_train
gs._y_train = y_train
gs._X_test = X_test
gs._y_test = y_test

# Run with custom representations
results = gs.run_grid_search(
    representations={"custom_features": X_train},
    preprocessing='standard'
)
```

## Adaptive Grid Search

Perform multi-stage hyperparameter optimization that refines the search space:

```python
from polyglotmol.models import AdaptiveGridSearch
from sklearn.ensemble import RandomForestRegressor

# Initial broad parameter grid
initial_params = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 20]
}

# Create adaptive search
adaptive_gs = AdaptiveGridSearch(
    output_dir="./adaptive_results",
    task_type="regression",
    model_name="Random Forest",
    representation_name="morgan_fp",
    initial_param_grid=initial_params,
    metric="neg_mean_squared_error",
    model_instance=RandomForestRegressor(random_state=42),
    higher_is_better=True,
    max_stages=3  # Refine parameters over 3 stages
)

# Extract features from dataset
features = dataset.get_features("morgan_fp_r2_1024")
X = np.vstack(features.iloc[:, 0].values)
y = dataset.get_labels("activity").values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Run adaptive search
best_model, best_params, best_score = adaptive_gs.run(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

print(f"Best parameters after refinement: {best_params}")
print(f"Best CV score: {best_score:.4f}")
```

## Progress Tracking

Grid search provides real-time progress updates:

```python
# Progress is automatically logged during execution
# Example output:
# INFO - Starting grid search with 20 models and 3 representations (1200 total combinations)
# INFO - Processing representation: morgan_fp_r2_1024
# INFO - Progress: 50/1200 (4.2%), Errors: 0, Est. remaining: 5h 23m 15s
# INFO - Best Random Forest with morgan_fp_r2_1024: RMSE=0.2345, R²=0.8765

# Access progress summary after completion
if gs.progress_tracker:
    summary = gs.progress_tracker.summary()
    print(f"Total time: {summary['elapsed_time']}")
    print(f"Completed: {summary['completed']}/{summary['total_combinations']}")
    print(f"Errors: {summary['errors']}")
    
    # Average times per model
    for model, avg_time in summary['model_avg_times'].items():
        print(f"{model}: {avg_time:.2f}s average")
```

## Convenience Functions

Quick one-liners for common tasks:

```python
from polyglotmol.models import run_regression_grid_search, run_classification_grid_search

# Regression with defaults
gs = run_regression_grid_search(
    dataset=dataset,
    feature_column="morgan_fp_r2_1024",
    target_column="activity",
    output_dir="./quick_regression"
)

# Classification with custom settings
gs = run_classification_grid_search(
    dataset=dataset,
    feature_column="ecfp4_1024",
    target_column="activity_class",
    test_size=0.25,
    cv=10,
    models=["Random Forest", "SVM", "XGBoost"],
    output_dir="./quick_classification"
)
```

## Tips and Best Practices

1. **Start with a subset**: Test on a small subset of data first to estimate runtime
2. **Use parallel processing**: Set `n_jobs=-1` to use all CPU cores
3. **Choose appropriate metrics**: Use `comparison_metric` to optimize for your specific goal
4. **Save models**: Keep `save_models=True` to reload the best model later
5. **Monitor progress**: Check the log file in the output directory for detailed progress

## API Reference

For detailed API documentation, see {doc}`/api/models/grid_search`.