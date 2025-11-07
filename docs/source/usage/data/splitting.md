# Dataset Splitting Strategies

Comprehensive guide to data splitting and cross-validation strategies in PolyglotMol.

## Overview

PolyglotMol provides flexible, professional-grade data splitting strategies for molecular machine learning. The splitting system is designed to:

- **Ensure fair model comparison** through consistent random seeds
- **Support multiple splitting strategies** for different use cases
- **Handle both classification and regression** with appropriate techniques
- **Maintain reproducibility** across different runs

```{admonition} Key Feature
:class: tip

All splitting strategies use **fixed random seeds** by default, ensuring complete reproducibility of model evaluations across different runs and users.
```

## Quick Start

### Basic Usage

```python
from polyglotmol.models import universal_screen
from polyglotmol.data import MolecularDataset

# Load your dataset
dataset = MolecularDataset.from_csv("molecules.csv",
                                   input_column="SMILES",
                                   label_columns=["activity"])

# Use default splitting (train_test with 80/20 split)
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    test_size=0.2,          # 20% for testing
    cv_folds=5,             # 5-fold cross-validation
    random_state=42         # Fixed random seed
)
```

## Supported Splitting Strategies

PolyglotMol supports 5 different splitting strategies, each suited for specific scenarios:

| Strategy | Use Case | Train/Val/Test | Best For |
|----------|----------|----------------|----------|
| `train_test` | Standard screening | 80% / — / 20% | Most common scenarios |
| `train_val_test` | With hyperparameter optimization | 70% / 15% / 15% | Large datasets with HPO |
| `nested_cv` | Unbiased HPO performance | Nested CV | Academic research |
| `cv_only` | Small datasets | CV only | < 100 samples |
| `user_provided` | Custom splits | User-defined | Scaffold, temporal splits |

### 1. Train/Test Split (Default)

The standard two-way split used for most screening tasks.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",  # Default
    test_size=0.2,                # 20% test set
    cv_folds=5,                   # 5-fold CV on training set
    random_state=42
)
```

#### How It Works

```
Dataset (1000 samples)
    ↓
train_test_split(test_size=0.2, random_state=42)
    ├─→ Training Set: 800 samples (80%)
    │      ↓
    │   5-Fold Cross-Validation
    │   ├─→ Fold 1: train 640, val 160
    │   ├─→ Fold 2: train 640, val 160
    │   ├─→ Fold 3: train 640, val 160
    │   ├─→ Fold 4: train 640, val 160
    │   └─→ Fold 5: train 640, val 160
    │      ↓
    │   Final model: trained on all 800 samples
    │
    └─→ Test Set: 200 samples (20%)
           ↓
        Final evaluation
```

#### Implementation Details

**Code Location**: `src/polyglotmol/models/api/core/splitting/strategies.py:26-84`

```python
def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Standard train/test split.

    For classification, uses StratifiedShuffleSplit to maintain class balance.
    For regression, uses regular train_test_split with shuffling.
    """
    if stratify is not None:
        # Stratified split for classification
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, stratify))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Regular split for regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'split_type': 'train_test'
    }
```

### 2. Train/Val/Test Split

Three-way split for scenarios involving hyperparameter optimization.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_val_test",
    test_size=0.15,      # 15% test set
    val_size=0.15,       # 15% validation set
    random_state=42
)
```

#### How It Works

```
Dataset (1000 samples)
    ↓
First split: (train+val) vs test
    ├─→ Temp Set: 850 samples (85%)
    │      ↓
    │   Second split: train vs val
    │   ├─→ Training Set: 700 samples (70%)
    │   └─→ Validation Set: 150 samples (15%)
    │
    └─→ Test Set: 150 samples (15%)
```

#### Use Cases

- **Hyperparameter optimization**: Use validation set to tune hyperparameters
- **Model selection**: Choose best model architecture on validation set
- **Large datasets**: When you have enough data (>5000 samples) for three-way split

#### Implementation

**Code Location**: `splitting/strategies.py:86-169`

```python
def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Three-way split: train / validation / test.

    Best for HPO when you have sufficient data.
    """
    # First split: (train+val) vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp if stratify is not None else None,
        shuffle=True
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'split_type': 'train_val_test'
    }
```

### 3. Nested Cross-Validation

Provides unbiased performance estimates for hyperparameter optimization.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,     # Outer CV for performance estimation
    inner_cv_folds=3,     # Inner CV for hyperparameter tuning
    random_state=42
)
```

#### How It Works

```
Dataset (1000 samples)
    ↓
Outer CV (5 folds) - for performance estimation
    ├─→ Fold 1: dev 800, test 200
    │      ↓
    │   Inner CV (3 folds on dev set) - for HPO
    │   ├─→ Inner Fold 1: train 533, val 267
    │   ├─→ Inner Fold 2: train 533, val 267
    │   └─→ Inner Fold 3: train 534, val 266
    │      ↓
    │   Best hyperparameters → test on outer test (200)
    │
    ├─→ Fold 2: dev 800, test 200
    │   (repeat inner CV...)
    ...
    └─→ Fold 5: dev 800, test 200
           ↓
        Average performance across 5 outer folds
```

#### Use Cases

- **Academic research**: Unbiased performance estimation for publications
- **Model comparison**: Fair comparison when HPO is involved
- **Small-to-medium datasets**: Maximize data utilization

#### Implementation

**Code Location**: `splitting/strategies.py:215-284`

```python
def get_nested_cv_splitter(
    n_samples: int,
    outer_cv_folds: int = 5,
    inner_cv_folds: int = 3,
    random_state: int = 42,
    is_classification: bool = False
) -> Dict[str, Any]:
    """
    Get nested cross-validation splitters.

    Nested CV provides unbiased performance estimates when doing HPO.
    - Outer CV: For performance estimation
    - Inner CV: For hyperparameter tuning
    """
    if is_classification:
        outer_cv = StratifiedKFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state
        )
        inner_cv = StratifiedKFold(
            n_splits=inner_cv_folds,
            shuffle=True,
            random_state=random_state + 1  # Different seed
        )
    else:
        outer_cv = KFold(
            n_splits=outer_cv_folds,
            shuffle=True,
            random_state=random_state
        )
        inner_cv = KFold(
            n_splits=inner_cv_folds,
            shuffle=True,
            random_state=random_state + 1
        )

    return {
        'outer_cv': outer_cv,
        'inner_cv': inner_cv,
        'split_type': 'nested_cv'
    }
```

### 4. CV-Only Strategy

Pure cross-validation without a separate test set, for small datasets.

#### Configuration

```python
results = universal_screen(
    dataset=small_dataset,  # < 100 samples
    target_column="activity",
    split_strategy="cv_only",
    cv_folds=5,
    random_state=42
)
```

#### How It Works

```
Dataset (80 samples)
    ↓
5-Fold Cross-Validation (no separate test set)
    ├─→ Fold 1: train 64, val 16
    ├─→ Fold 2: train 64, val 16
    ├─→ Fold 3: train 64, val 16
    ├─→ Fold 4: train 64, val 16
    └─→ Fold 5: train 64, val 16
           ↓
        Average CV score as final metric
```

#### Use Cases

- **Small datasets**: < 100 samples where test set would be too small
- **Maximum data utilization**: Every sample used for both training and validation
- **Exploratory analysis**: Quick performance estimates

```{warning}
CV-only strategy doesn't provide a truly independent test set. Performance estimates may be optimistically biased. Use only when dataset size prohibits train/test split.
```

### 5. User-Provided Splits

Custom splitting for specialized scenarios like scaffold or temporal splits.

#### Configuration

```python
# Example: Scaffold-based split
from rdkit.Chem.Scaffolds import MurckoScaffold

# Compute scaffolds for your molecules
scaffolds = [
    MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    for mol in dataset.molecules
]

# Create custom train/test indices
# (Example implementation - you would implement scaffold-based logic)
train_indices, test_indices = custom_scaffold_split(scaffolds, test_size=0.2)

# Pass to universal_screen
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="user_provided",
    user_splits={
        'train_indices': train_indices,
        'test_indices': test_indices
    }
)
```

#### Use Cases

- **Scaffold splits**: Ensure test set contains novel chemical scaffolds
- **Temporal splits**: Time-based train/test division
- **Stratified splits**: Custom stratification logic
- **External test sets**: Pre-defined validation datasets

## Cross-Validation Details

### Adaptive CV Configuration

PolyglotMol automatically adjusts cross-validation based on dataset size and task type.

#### Automatic Fold Adjustment

**Code Location**: `evaluation/evaluator.py:288-299`

```python
def _cross_validate(self, model, X, y):
    """Perform cross-validation with automatic fold adjustment."""

    # Validate cv_folds parameter
    cv_folds = self.config.cv_folds
    n_samples = len(y)

    # Ensure we don't have more folds than samples
    if cv_folds > n_samples:
        logger.warning(f"cv_folds={cv_folds} > n_samples={n_samples}, "
                      f"using cv_folds={n_samples}")
        cv_folds = n_samples

    # Minimum 2 folds required
    if cv_folds < 2:
        logger.warning(f"cv_folds={cv_folds} invalid, using cv_folds=2")
        cv_folds = 2
```

#### Classification vs Regression

**Code Location**: `evaluation/evaluator.py:302-315`

```python
# Create CV splitter based on task type
if self.config.task_type in [TaskType.CLASSIFICATION,
                             TaskType.BINARY_CLASSIFICATION]:
    # Use StratifiedKFold for classification
    cv_splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=self.config.random_state
    )
else:
    # Use KFold for regression
    cv_splitter = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=self.config.random_state
    )
```

### Stratified Sampling

For classification tasks, PolyglotMol automatically uses **stratified sampling** to maintain class balance across folds.

#### Benefits

- **Maintains class distribution** in each fold
- **Prevents fold-to-fold variance** due to class imbalance
- **More reliable CV scores** for imbalanced datasets

#### Example

```python
# Classification dataset with imbalanced classes
# Class 0: 800 samples, Class 1: 200 samples

# Without stratification (bad):
# Fold 1 might have: Class 0: 195, Class 1: 5 (95% vs 5%)
# Fold 2 might have: Class 0: 165, Class 1: 35 (83% vs 17%)

# With StratifiedKFold (good):
# All folds maintain: Class 0: 160, Class 1: 40 (80% vs 20%)
```

## Choosing the Right Strategy

### Decision Tree

```
How much data do you have?
    │
    ├─ < 100 samples
    │      └─→ Use cv_only (maximize data usage)
    │
    ├─ 100-500 samples
    │      ├─ Need HPO? → nested_cv
    │      └─ Otherwise → train_test (test_size=0.3)
    │
    ├─ 500-5000 samples
    │      ├─ Need HPO? → train_val_test
    │      └─ Otherwise → train_test (test_size=0.2)
    │
    └─ > 5000 samples
           ├─ Need HPO? → train_val_test
           ├─ Academic study? → nested_cv
           └─ Otherwise → train_test (test_size=0.15)
```

### Recommended Configurations

| Scenario | Strategy | test_size | cv_folds | Rationale |
|----------|----------|-----------|----------|-----------|
| Quick screening (any size) | `train_test` | 0.2 | 3 | Fast, reasonable estimates |
| Standard screening (>500) | `train_test` | 0.2 | 5 | Balanced, industry standard |
| Small dataset (<100) | `cv_only` | — | 5 | Maximum data utilization |
| Large dataset (>10K) | `train_test` | 0.1 | 3 | Efficient, large test set |
| HPO required (>1K) | `train_val_test` | 0.15 | 0.15 | Dedicated validation set |
| Research/publication | `nested_cv` | — | 5/3 | Unbiased performance |
| Scaffold split needed | `user_provided` | Custom | 5 | Domain-specific |

## Reproducibility

### Fixed Random Seeds

All splitting strategies use fixed random seeds by default to ensure reproducibility.

**Code Location**: `models/api/core/base.py:215-217`

```python
@dataclass
class ScreeningConfig:
    """Configuration for model screening."""

    cv_folds: int = 5            # Cross-validation folds
    test_size: float = 0.2       # Test set proportion
    random_state: int = 42       # Fixed random seed
```

### Guaranteed Reproducibility

```python
# Run 1
results1 = universal_screen(
    dataset=dataset,
    target_column="activity",
    random_state=42  # Fixed
)

# Run 2 (different session)
results2 = universal_screen(
    dataset=dataset,
    target_column="activity",
    random_state=42  # Same seed
)

# Guarantee: results1 and results2 will have identical splits
assert (results1['test_indices'] == results2['test_indices']).all()
```

### What's Reproducible

✅ **Guaranteed reproducible** (with same `random_state`):
- Train/test split indices
- Cross-validation fold assignments
- Model training (if model uses same seed)
- Final test scores

⚠️ **May vary slightly**:
- Training time (system load dependent)
- Memory usage (Python GC behavior)

## Advanced Usage

### Custom Splitting Logic

If you need specialized splitting logic not covered by the built-in strategies, use the `user_provided` strategy:

```python
from polyglotmol.models.api.core.splitting import validate_user_splits

# Your custom splitting logic
def my_custom_split(dataset, test_ratio=0.2):
    """Custom split based on molecular properties."""

    # Example: Split by molecular weight
    mol_weights = [mol.GetDescriptors()['MolWt']
                   for mol in dataset.molecules]

    # Sort by molecular weight
    sorted_indices = np.argsort(mol_weights)
    n_test = int(len(dataset) * test_ratio)

    # Heaviest molecules in test set
    test_indices = sorted_indices[-n_test:]
    train_indices = sorted_indices[:-n_test]

    return train_indices, test_indices

# Create splits
train_idx, test_idx = my_custom_split(dataset, test_ratio=0.2)

# Validate splits (checks for overlaps, coverage, etc.)
validated_splits = validate_user_splits(
    X=dataset.features.values,
    y=dataset.labels.values,
    user_splits={
        'train_indices': train_idx,
        'test_indices': test_idx
    }
)

# Use in screening
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="user_provided",
    user_splits=validated_splits
)
```

### Combining Multiple Strategies

For comprehensive model evaluation, you can run multiple splitting strategies and compare:

```python
strategies = ['train_test', 'nested_cv', 'cv_only']
results_by_strategy = {}

for strategy in strategies:
    results_by_strategy[strategy] = universal_screen(
        dataset=dataset,
        target_column="activity",
        split_strategy=strategy,
        random_state=42
    )

# Compare performance across strategies
for strategy, results in results_by_strategy.items():
    print(f"{strategy}: R² = {results['best_model']['test_r2']:.3f}")
```

## Best Practices

```{admonition} Splitting Best Practices
:class: tip

**Always:**
- Use fixed `random_state` for reproducibility
- Choose `test_size` based on dataset size (see table above)
- Use stratification for classification tasks (automatic)
- Validate custom splits before use

**Never:**
- Use test set for hyperparameter tuning
- Peek at test set during model development
- Use different splits for comparing models
- Ignore warnings about insufficient samples per fold
```

## Performance Considerations

### Memory Efficiency

Different strategies have different memory footprints:

| Strategy | Memory Usage | Speed | Best For |
|----------|--------------|-------|----------|
| `train_test` | Low | Fast | Large datasets |
| `cv_only` | Medium | Medium | Small datasets |
| `train_val_test` | Low | Fast | Large datasets with HPO |
| `nested_cv` | High | Slow | Medium datasets, research |

### Computational Cost

```python
# Fastest: Simple train/test with 3-fold CV
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="train_test",
    cv_folds=3  # ~2x faster than 5-fold
)

# Slowest: Nested CV with many folds
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="nested_cv",
    outer_cv_folds=5,
    inner_cv_folds=5  # 25 total model fits per model type
)
```

## Related Topics

- {doc}`methodology` - Complete evaluation methodology documentation
- {doc}`../models/screening` - Model screening API reference
- {doc}`dataset` - Dataset management guide

## References

- [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Nested Cross-Validation Explained](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
- [Stratified Sampling](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

## Summary

```{admonition} Key Takeaways
:class: tip

1. **5 Splitting Strategies**: train_test (default), train_val_test, nested_cv, cv_only, user_provided
2. **Automatic Stratification**: Classification tasks use StratifiedKFold automatically
3. **Fixed Random Seeds**: `random_state=42` ensures complete reproducibility
4. **Adaptive Configuration**: CV folds automatically adjusted based on dataset size
5. **Flexible Integration**: Easy to plug in custom splitting logic via `user_provided`
```
