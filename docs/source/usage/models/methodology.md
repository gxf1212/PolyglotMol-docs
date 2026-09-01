# Evaluation Methodology

Complete guide to MolBlender's data splitting, cross-validation, and model evaluation protocols.

## Overview

MolBlender follows rigorous machine learning best practices to ensure fair model comparison and reproducible results. This document provides a detailed explanation of:

- How training and test sets are created
- How 5-fold cross-validation is performed
- How models are trained and evaluated
- How to interpret metrics in the Dashboard
- Code references for understanding the implementation

```{admonition} Key Principle
:class: tip

All models are evaluated on the **same test set** with **fixed random seeds** to ensure fair comparison and reproducibility.
```

## Data Splitting Strategy

```{admonition} Advanced Splitting Strategies
:class: seealso

This section covers the **internal train/test splitting** used during model screening. For comprehensive dataset splitting strategies (scaffold split, temporal split, etc.) to prepare your data **before** running screening, see {doc}`/usage/data/splitting`.
```

### Train/Test Split

MolBlender uses sklearn's `train_test_split` to divide your dataset into training and test sets.

#### Default Configuration

```python
# Default parameters in ScreeningConfig (base.py:215-217)
cv_folds: int = 5            # Cross-validation folds
test_size: float = 0.2       # Test set proportion (20%)
random_state: int = 42       # Random seed for reproducibility
```

#### Implementation Details

**Code Location**: `src/molblender/models/api/screening_engine/data_handler.py`

```python
def split_data(self, X: np.ndarray, y: np.ndarray, stratify=None):
    splitter = DataSplitter(
        strategy=self.config.split_strategy,
        test_size=self.config.test_size,
        cv_folds=self.config.cv_folds,
        random_state=self.config.random_state,
        # strategy-specific options are forwarded from ScreeningConfig
    )
    return splitter.split(
        X=X, y=y, dataset=self.dataset, smiles=self.smiles_list,
        is_classification=is_classification_task(self.config.task_type),
        stratify=stratify,
    )
```

The configured strategy, rather than this method, determines whether the
result is a holdout split, a CV-only split, nested CV, or a chemistry-aware
split. The returned keys therefore depend on that strategy.

#### Numerical Example

For a dataset with **1000 molecules**:

```
Original Dataset: 1000 samples
    ↓
split_data(test_size=0.2, random_state=42)
    ├─→ Training Set: 800 samples (80%)
    └─→ Test Set: 200 samples (20%)
```

#### Stratified Splitting for Classification

For classification tasks, MolBlender automatically uses **stratified sampling** to maintain class balance:

**Code Location**: `data_handler.py:108-117`

```python
if stratify is None and self.config.task_type in [TaskType.CLASSIFICATION, ...]:
    unique_labels = np.unique(y)
    if len(unique_labels) < len(y) * 0.5:  # Not too many classes
        stratify = y  # Enable stratification

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=self.config.test_size,
        random_state=self.config.random_state
    )
```

### Reproducibility Guarantee

The `random_state=42` parameter ensures that:
- The same dataset will always be split the same way
- Results are reproducible across different runs
- Different users get identical train/test splits

```{admonition} Shared Split Plan
:class: tip

All representations share the same train/test and CV fold indices via a `SplitPlan`. Indices are computed once (from a reference representation) and applied to every representation, ensuring consistent splits. If a representation has fewer samples than the index set, only the overlapping molecules are used.
```

## Cross-Validation Protocol

### 5-Fold Cross-Validation

Cross-validation is performed **only on the training set** to estimate model performance without touching the test set.

#### Why Cross-Validation?

- **Reduces overfitting**: Tests model on multiple train/validation splits
- **Better performance estimation**: Averages over 5 different validation sets
- **Efficient use of data**: Uses all training data for both training and validation

#### Implementation Details

**Code Location**: `src/molblender/models/api/screening_engine/evaluation/cross_validation.py`

```python
def perform_cross_validation(model, X, y, config, model_requires_scaling=False):
    cv_splitter = create_cv_splitter(
        task_type=config.task_type,
        cv_folds=config.cv_folds,
        random_state=config.random_state,
        stratify_labels=config.stratify,
    )
    # The implementation validates feasible fold counts and wraps scaling in
    # the per-fold pipeline when required, preventing preprocessing leakage.
    return cross_val_score(
        maybe_wrap_in_scaling_pipeline(model, model_requires_scaling),
        X, y, cv=cv_splitter,
        scoring=get_sklearn_scoring(config.task_type, config.primary_metric),
    )
```

**Cross-Validation Call**: `evaluator.py` calls `perform_cross_validation()`

```python
# Called during model evaluation
cv_scores = perform_cross_validation(model, X_train, y_train, config)
```

#### How sklearn Splits the Data

When you call `cross_val_score(model, X_train, y_train, cv=cv_splitter)`, sklearn internally:

```python
# Pseudo-code for sklearn's internal logic
# MolBlender creates KFold with fixed random_state
kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # ✅ Fixed random_state

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    # Split training data into CV train and validation
    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    # Train model on CV training set
    cloned_model = clone(model)
    cloned_model.fit(X_cv_train, y_cv_train)

    # Evaluate on CV validation set
    score = scoring(cloned_model, X_cv_val, y_cv_val)
    scores.append(score)

return np.array(scores)  # [0.85, 0.87, 0.84, 0.86, 0.88]
```

#### Numerical Example

For a training set with **800 samples**:

```
Training Set: 800 samples
    ↓
5-Fold Cross-Validation
    ├─→ Fold 1: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score1 = 0.85
    ├─→ Fold 2: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score2 = 0.87
    ├─→ Fold 3: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score3 = 0.84
    ├─→ Fold 4: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score4 = 0.86
    └─→ Fold 5: CV_train 640 samples (80%) | CV_val 160 samples (20%) → score5 = 0.88
        ↓
Mean CV Score = (0.85 + 0.87 + 0.84 + 0.86 + 0.88) / 5 = 0.86
Std CV Score = 0.015
```

```{admonition} Reproducibility
:class: tip

MolBlender now uses `KFold(random_state=42)` for cross-validation, ensuring **complete reproducibility** of both CV scores and test scores across different runs.
```

## Model Training and Evaluation

### Final Model Training

After cross-validation, the model is trained on the **entire training set** to maximize performance.

**Code Location**: `models/api/screening_engine/evaluation/evaluator.py`

```python
# Train final model using ALL training data
execution_context = ExecutionContext.from_screening_config(config)

with execution_context.timeout_context(timeout=train_timeout):
    n_samples = X_train.shape[0]  # 800 samples
    logger.debug(f"Training {model_name} on {n_samples} samples")

    # Train on full training set (not just one fold)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
```

MolBlender now routes timeout and runtime policy through `molblender.infrastructure.ExecutionContext` rather than the removed standalone `timeout_context` helper.

### Test Set Evaluation

The trained model is evaluated on the **held-out test set**.

**Code Location**: `evaluator.py:192-210`

```python
# Predict on test set
test_pred = model.predict(X_test)   # 200 predictions
train_pred = model.predict(X_train) # 800 predictions (for diagnostics)

# Compute test metrics
test_metrics = compute_metrics(y_test, test_pred)
# test_metrics = {'r2': 0.87, 'rmse': 0.52, 'mae': 0.41, ...}
```

### Metrics Explained

MolBlender reports **two types of metrics**:

| Metric Type | Source | Purpose | Dashboard Display |
|------------|--------|---------|-------------------|
| **CV Scores** | 5-fold cross-validation on training set | Estimate model robustness | Mean ± Std in tables |
| **Test Scores** | Final evaluation on test set | True generalization performance | **Primary metrics** |

#### What Dashboard Shows

```{admonition} Metrics in Results and Selection
:class: important

Each result stores a `primary_metric` field (the **test set score**) and CV scores. The Dashboard displays the test score as the main metric for each model-representation combination.

However, **model selection** (choosing the "best" model) is based on **CV or HPO scores**, not test scores. The screening engine uses `rank_final_results()` which ranks by CV/HPO scores and never falls back to test scores for winner selection (unless `allow_test_fallback=True` is explicitly set). This prevents test set contamination from influencing model choice.

In other words:
- **What you see**: test scores are the headline numbers for each model
- **How the winner is chosen**: CV/HPO scores determine the best model
```

**Example Result**:

```
Random Forest + Morgan FP:
├─ Test R² = 0.87     ← Displayed as primary_metric (test set evaluation)
├─ CV R² = 0.86 ± 0.015 ← Used for model ranking and selection
├─ Test RMSE = 0.52
└─ Training Time = 12.3s
```

The CV score (0.86) is what determines the model's rank relative to other candidates. The test score (0.87) provides an independent estimate of real-world performance.

## Complete Evaluation Workflow

### Visual Overview

```
Original Dataset: 1000 molecules
    ↓
[data_handler.split_data() - data_handler.py:120]
    ├─→ X_train, y_train: 800 samples (80%)
    │       ↓
    │   [evaluator._cross_validate() - evaluator.py:139]
    │       ↓
    │   [sklearn.cross_val_score(cv=5) - evaluator.py:302]
    │   ├─→ Fold 1: CV_train 640, CV_val 160 → score1
    │   ├─→ Fold 2: CV_train 640, CV_val 160 → score2
    │   ├─→ Fold 3: CV_train 640, CV_val 160 → score3
    │   ├─→ Fold 4: CV_train 640, CV_val 160 → score4
    │   └─→ Fold 5: CV_train 640, CV_val 160 → score5
    │       ↓
    │   cv_scores = [0.85, 0.87, 0.84, 0.86, 0.88]
    │   mean_cv = 0.86, std_cv = 0.015
    │       ↓
    │   [model.fit(X_train, y_train) - evaluator.py:167]
    │   Final model trained on ALL 800 training samples
    │
    └─→ X_test, y_test: 200 samples (20%)
            ↓
        [model.predict(X_test)]
        test_pred = [pred1, pred2, ..., pred200]
            ↓
        [compute_metrics(y_test, test_pred)]
        test_score = 0.87 ← stored as `primary_metric` in results
            ↓
        [rank_final_results() — ranks by CV/HPO scores]
        selection_source = "cv_score" or "hpo_score"
        best_model ← chosen by CV/HPO ranking, NOT test score
```

### Step-by-Step Execution

1. **Data Loading**: Load dataset with molecules and target values
2. **Feature Extraction**: Compute molecular representations (fingerprints, descriptors, etc.)
3. **Train/Test Split**: Divide data into 80% train (800) and 20% test (200)
4. **Cross-Validation**: 5-fold CV on training set to get CV scores
5. **Final Training**: Train model on entire training set (800 samples)
6. **Test Evaluation**: Predict on test set (200 samples) and compute test scores
7. **Result Storage**: Save all metrics to SQLite database
8. **Model Selection**: Rank by CV/HPO scores via `rank_final_results()`; test scores are stored but not used for winner selection
9. **Dashboard Display**: Show test scores (`primary_metric`) as headline numbers; CV/HPO scores drive the ranking

## Best Practices

### Choosing test_size

The `test_size` parameter controls the train/test split ratio. Choose based on your dataset size:

| Dataset Size | Recommended test_size | Train Samples | Test Samples | Rationale |
|-------------|----------------------|---------------|--------------|-----------|
| < 500 | `0.3` (30%) | 350 | 150 | Need sufficient test samples for reliable evaluation |
| 500 - 5000 | `0.2` (20%) | 4000 | 1000 | Default - balanced split |
| 5000 - 50000 | `0.15` (15%) | 42500 | 7500 | More data for training |
| > 50000 | `0.1` (10%) | 90000 | 10000 | Large test set still provides good statistics |

**Example**:

```python
# For a small dataset with 300 molecules
results = universal_screen(
    dataset=small_dataset,
    target_column="activity",
    test_size=0.3,  # Use 30% for testing (90 molecules)
    cv_folds=3      # Fewer folds to ensure enough data per fold
)

# For a large dataset with 100,000 molecules
results = universal_screen(
    dataset=large_dataset,
    target_column="solubility",
    test_size=0.1,  # Only 10% for testing (10,000 molecules)
    cv_folds=5      # Standard 5-fold CV
)
```

### Choosing cv_folds

The `cv_folds` parameter controls the number of cross-validation folds:

| Dataset Size | Recommended cv_folds | CV Train | CV Val | Rationale |
|-------------|---------------------|----------|--------|-----------|
| < 100 | `3` | 67 | 33 | Avoid too-small validation sets |
| 100 - 1000 | `5` (default) | 640 | 160 | Standard choice, good balance |
| 1000 - 10000 | `5` or `3` | 6400 | 1600 | Use 3 for speed |
| > 10000 | `3` | 24000 | 8000 | Faster, minimal performance loss |

**Speed vs Accuracy Trade-off**:

```python
# Fast screening for exploration (2-3x faster)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=3  # 3-fold CV
)

# Thorough screening for final evaluation (default)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=5  # 5-fold CV
)

# Very thorough (research/publication, 2x slower)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    cv_folds=10  # 10-fold CV
)
```

### Ensuring Fair Comparison

To ensure all models are compared fairly:

1. **Use the same random_state** (default: 42)
2. **Use the same test_size** for all screening runs
3. **Use the same cv_folds** for all models

```python
# Good: All models compared on same splits
config_params = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

results1 = universal_screen(dataset, "activity", **config_params)
results2 = compare_models(dataset, "activity", "morgan_fp", **config_params)
results3 = compare_representations(dataset, "activity", "random_forest", **config_params)
```

```{admonition} Ensuring Consistent Splits
:class: tip

All representations share the same split indices via `SplitPlan`. For critical comparisons, verify that all representations are computed for the same molecule set to ensure identical sample coverage.
```

## Splitting Architecture

### Shared Split Plan

MolBlender computes the train/test (and CV fold) indices **once** via a `SplitPlan`, then applies the same indices to all representations. This guarantees that every representation trains and evaluates on the same sample indices.

```python
# Internal flow (simplified)
split_plan = _extract_split_plan(config, X=reference_X, y=y)  # indices computed once
for repr_name, X in representations.items():
    X_train, X_test, y_train, y_test = _apply_shared_split(split_plan, X, y)
    # All representations share the same train/test indices
```

When representations have different sample counts (e.g., due to missing values), the shared indices still apply — only molecules present in both the representation and the index set are used. The `SplitPlan` records a cohort fingerprint to detect when sample sets differ.

### CV Random State ✅

MolBlender creates `KFold`/`StratifiedKFold` objects with `random_state=42` before passing to `cross_val_score`, ensuring both CV scores and test scores are fully reproducible.

## Code Reference Index

Quick reference to implementation modules (line numbers change frequently):

| Functionality | Module | Description |
|--------------|--------|-------------|
| **Configuration** | `screening_engine/base.py` | `ScreeningConfig` with `cv_folds`, `test_size`, `random_state` |
| **Split Plan** | `screening_runtime/split_plan.py` | `_extract_split_plan` / `_apply_shared_split` — compute indices once, apply to all representations |
| **CV Splitter** | `screening_engine/evaluation/cross_validation.py` | `create_cv_splitter` — KFold / StratifiedKFold with `random_state=42` |
| **CV Execution** | `screening_engine/evaluation/cross_validation.py` | `perform_cross_validation` — `cross_val_score` with scoring |
| **Model Ranking** | `screening_engine/evaluation/ranking.py` | `rank_final_results` / `selection_score` — rank by CV/HPO scores |
| **Final Selection** | `screening_runtime/run.py` | `_select_best_result` — calls `rank_final_results`, never falls back to test score |

## Large Dataset Prediction

MolBlender supports efficient prediction on large datasets with automatic batch processing and memory management:

### Inference Batch Size Control

Deep learning models (VAE, CNN, Transformer) use batch processing for inference to prevent GPU memory overflow:

```python
from molblender.models.api import universal_screen

# Large dataset prediction
results = universal_screen(
    dataset=large_dataset,
    target_column="activity",
    cv_folds=3,           # Reduce CV folds
    test_size=0.1,         # Smaller test set
    enable_heavy_gpu_scheduler=True,  # Enable GPU scheduling
    heavy_jobs_per_gpu=1,  # One heavy task per GPU for large datasets
    train_timeout=600     # Longer timeout for large models
)
```

**Batch Processing Note**:
Deep learning models (VAE, CNN, Transformer) use batch processing internally with default batch sizes defined in the model catalog:

| Model Type | Default Batch Size | Note |
|-----------|-------------------|------|
| VAE | 16-32 | Configurable in model catalog |
| CNN | 16 | Smaller batches for matrix inputs |
| Transformer | 16 | Configurable in model catalog |
| Traditional ML | N/A | No batch processing needed |

### GPU Memory Optimization

The system automatically handles GPU memory constraints:

**Memory Management Features**:
- Automatic batch size adjustment on CUDA OOM
- GPU memory cleanup after each batch
- CPU fallback when GPU is unavailable

For very large datasets, enable the GPU scheduler to prevent OOM errors:

```python
# Recommended settings for large datasets (10k+ molecules)
results = universal_screen(
    dataset=large_dataset,
    target_column="activity",
    cv_folds=3,           # Reduce CV folds
    test_size=0.1,         # Smaller test set
    enable_heavy_gpu_scheduler=True,  # Enable GPU scheduling
    heavy_jobs_per_gpu=1,  # One heavy task per GPU for large datasets
    train_timeout=600     # Longer timeout for large models
)
```

**Dataset Size Guidelines**:

| Dataset Size | Batch Size | CV Folds | Test Size |
|-------------|-----------|----------|-----------|
| < 1,000 | 256 (default) | 5 | 0.2 |
| 1,000 - 10,000 | 256 | 5 | 0.2 |
| 10,000 - 50,000 | 128-256 | 3 | 0.1 |
| > 50,000 | 64-128 | 3 | 0.05-0.1 |

## Further Reading

- {doc}`screening` - Complete API reference for screening functions
- {doc}`results` - Understanding and exporting screening results
- {doc}`../dashboard/metrics` - Dashboard metric interpretation
- [sklearn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) - sklearn documentation

## Summary

```{admonition} Key Takeaways
:class: tip

1. **Train/Test Split**: 80/20 by default, fixed with `random_state=42`; indices computed once via SplitPlan and shared across representations
2. **Cross-Validation**: 5-fold CV performed **only on training set** with fixed `random_state=42`
3. **Final Model**: Trained on **full training set** (not just one fold)
4. **Test Evaluation**: Metrics computed on **held-out test set** and stored as `primary_metric`
5. **Model Selection**: Winner chosen by **CV/HPO scores** via `rank_final_results()` — test scores are never used for selection (unless `allow_test_fallback=True`)
6. **Dashboard Display**: Test scores shown as headline numbers; CV/HPO scores drive ranking
7. **Reproducibility**: ✅ Both CV scores and test scores are **fully reproducible**
```

For questions or issues about the evaluation methodology, please [open an issue on GitHub](https://github.com/gxf1212/MolBlender/issues).
