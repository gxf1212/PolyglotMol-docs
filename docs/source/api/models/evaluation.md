# Model Evaluation API

API reference for model evaluation functions and utilities.

## Overview

This module provides comprehensive evaluation tools for machine learning models, including visualization functions, metrics calculation, and performance analysis utilities.

## Evaluation Functions

### Regression Evaluation

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.evaluate_regression
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.calculate_regression_metrics
```

### Classification Evaluation

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.evaluate_classification
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.calculate_classification_metrics
```

## Visualization Functions

### Regression Plots

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_regression_results
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_residuals
```

### Classification Plots

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_classification_results
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_confusion_matrix
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_roc_curve
```

### Model Analysis

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_feature_importance
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_learning_curve
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.plot_model_comparison
```

## Utility Functions

### Cross-Validation

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.cross_validate_model
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.nested_cross_validation
```

### Model Selection

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.grid_search_evaluation
```

```{eval-rst}
.. autofunction:: polyglotmol.models.evaluation.compare_models
```

## Classes

### EvaluationReport

```{eval-rst}
.. autoclass:: polyglotmol.models.evaluation.EvaluationReport
   :members:
   :show-inheritance:
```

### ModelComparison

```{eval-rst}
.. autoclass:: polyglotmol.models.evaluation.ModelComparison
   :members:
   :show-inheritance:
```

## Configuration

### Plot Settings

```{eval-rst}
.. autoclass:: polyglotmol.models.evaluation.PlotConfig
   :members:
   :show-inheritance:
```

## Usage Examples

### Basic Regression Evaluation

```python
from polyglotmol.models.evaluation import evaluate_regression, plot_regression_results

# Evaluate regression model
metrics = evaluate_regression(y_true, y_pred)
print(f"R² Score: {metrics['r2']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")

# Create visualization
fig, ax = plot_regression_results(
    y_true=y_true,
    y_pred=y_pred,
    title="Model Performance",
    save_path="regression_results.png"
)
```

### Classification Evaluation

```python
from polyglotmol.models.evaluation import evaluate_classification, plot_classification_results

# Evaluate classification model
metrics = evaluate_classification(y_true, y_pred, y_pred_proba)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")

# Create comprehensive plot
fig, axes = plot_classification_results(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    class_names=["Active", "Inactive"],
    save_path="classification_results.png"
)
```

### Model Comparison

```python
from polyglotmol.models.evaluation import compare_models

# Compare multiple models
results = compare_models(
    models=[rf_model, svm_model, lr_model],
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    cv=5
)

# Display comparison
for model_name, metrics in results.items():
    print(f"{model_name}: R² = {metrics['test_r2']:.3f}")
```

## See Also

- {doc}`../../usage/models/evaluation` - Usage guide and examples
- {doc}`../representations/index` - Molecular representations
- {doc}`../data/index` - Data handling utilities