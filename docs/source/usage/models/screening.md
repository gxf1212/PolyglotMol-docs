# Screening Functions

Complete reference for MolBlender's screening functions - from quick evaluation to comprehensive multimodal screening.

## Overview

MolBlender provides multiple screening functions optimized for different use cases:

| **Function** | **Purpose** | **Models** | **Time** | **Use Case** |
|-------------|------------|-----------|---------|-------------|
| `simple_evaluate()` | Single model test | 1 model | <1 min | Quick baseline |
| `quick_screen()` | Fast essential screening | 5-10 | 2-5 min | Initial exploration |
| `thorough_screen()` | Accurate-model screening | 10-20 | 10-30 min | Balanced coverage |
| `interpretable_screen()` | Transparent models only | 5-10 | 2-5 min | Feature importance / regulatory |
| `screen_models()` | Full control via `model_corpus` | Custom | Varies | All of the above use this |
| `universal_screen()` | Multimodal comprehensive | 15-30 | 10-60 min | **Recommended default** |
| `compare_models()` | Model comparison | Custom | 5-15 min | Model selection |
| `compare_representations()` | Representation comparison | Custom | 5-15 min | Feature selection |

## universal_screen()

**The primary screening function** - automatically detects data modalities and selects compatible models.

### Basic Usage

```python
from molblender.models.api import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="activity"
)
```

### Recommended Usage with Config Objects

```python
from molblender.models.api import universal_screen
from molblender.models.api.screening_engine.configs import (
    CoreScreeningConfig, SplitConfig, ResourceConfig
)

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    screening_config=CoreScreeningConfig(task_type="regression", verbose=1),
    split_config=SplitConfig(strategy="scaffold", cv_folds=5),
    resource_config=ResourceConfig(max_cpu_cores=-1, max_workers_per_model=1),
)
```

### Compatibility: Legacy kwargs Passthrough

For backward compatibility, `universal_screen` accepts a flat set of legacy keyword arguments via `**legacy_kwargs`. These are validated and forwarded to the underlying screener.

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    cv_folds=5,
    test_size=0.2,
    primary_metric=None,
    combinations="auto",
    max_cpu_cores=-1,
)
```

### Parameters Explained

**Required Parameters**

`dataset`: `MolecularDataset`
: Input dataset with molecules and labels

`target_column`: `str`
: Name of the target variable column

**Configuration Objects (Recommended)**

The preferred API uses typed config objects for clarity:

`config_file`: `str` or `Path`, optional
: Path to a YAML/JSON config file that can specify all screening parameters at once.

`screening_config`: `CoreScreeningConfig`, optional
: Core task settings. Key fields:
  - `task_type`: `"regression"` or `"classification"`
  - `primary_metric`: evaluation metric (e.g., `"pearson_r"`, `"r2"`, `"rmse"` for regression; `"f1"`, `"accuracy"`, `"roc_auc"` for classification)
  - `verbose`: logging level (0/1/2)

`split_config`: `SplitConfig`, optional
: Data splitting strategy. Key fields:
  - `strategy`: `"scaffold"`, `"random"`, `"max_dissimilarity"`, or `"time"`
  - `cv_folds`: number of CV folds (default 5)
  - `test_size`: test set proportion (default 0.2)
  - `random_state`: seed for reproducibility (default 42)

`resource_config`: `ResourceConfig`, optional
: Compute resource settings. Key fields:
  - `max_cpu_cores`: total CPU cores (-1 = all available)
  - `max_workers_per_model`: per-estimator parallelism cap (default 1)
  - `execution_preference`: `"speed"`, `"memory"`, or `"balanced"` (default `"balanced"`)

`hpo_config`: `HPOConfig`, optional
: Hyperparameter optimization settings (see {doc}`hpo` for details).

`database_config`: `DatabaseConfig`, optional
: SQLite storage settings (path, enable/disable).

`weight_config`: `WeightConfig`, optional
: Sample weighting configuration for imbalanced datasets.

`fusion_config`: `FusionConfig`, optional
: Vector representation fusion configuration (dense 2D concatenation).

**Direct Parameters**

`modality_categories`: `List[str]`, optional
: Hierarchical category paths for representation selection. Auto-detected if `None`.
  ```python
  modality_categories=[
      "fingerprints/molecular",      # Molecular fingerprints
      "descriptors/physicochemical", # RDKit descriptors
      "sequential/language_model",   # Pre-trained embeddings
      "sequential/string",           # Raw SMILES for transformers
      "spatial/matrix",              # Adjacency/Coulomb matrices
      "image/2d"                     # 2D molecular images
  ]
  ```

Notes:
- `LANGUAGE_MODEL` representations are handled in their own modality path and are no longer double-processed through VECTOR routing.

`combinations`: `str` or `List`, default=`"auto"`
: Model selection strategy. Options:
  - `"auto"` **(recommended)** - Automatic selection based on modality
  - `"comprehensive"` - All compatible models including backup paths
  - `List[str]` - Specific model names: `["random_forest", "xgboost", "svm_rbf"]`
  - `List[Combination]` - Custom model-representation pairs

**Legacy kwargs (`**legacy_kwargs`)**

For backward compatibility, the following flat keyword arguments are still accepted and forwarded to the underlying screener. New code should prefer config objects.

| kwarg | Default | Description |
|-------|---------|-------------|
| `task_type` | `"regression"` | `"regression"` or `"classification"` |
| `primary_metric` | `None` | Auto-selected if not specified |
| `cv_folds` | `5` | Cross-validation folds |
| `test_size` | `0.2` | Test set proportion |
| `random_state` | `42` | Random seed |
| `verbose` | `1` | Logging verbosity (0/1/2) |
| `max_cpu_cores` | `-1` | CPU cores (-1 = all) |
| `max_workers_per_model` | `1` | Per-estimator parallelism cap |
| `execution_preference` | `"balanced"` | `"speed"`, `"memory"`, `"balanced"` |
| `enable_db_storage` | `False` | SQLite storage toggle |
| `db_path` | `None` | Database file path |
| `enable_hpo` | `False` | Two-stage HPO toggle |
| `hpo_stage` | `"coarse"` | `"coarse"` \| `"fine"` \| `"customized"` |
| `hpo_method` | `"grid"` | `"grid"` \| `"random"` |
| `top_n_for_hpo` | `5` | Models to optimize in Stage 2 |

**GPU Heavy-Task Scheduler** ⭐ **NEW**

MolBlender implements an **intelligent GPU scheduler** that distinguishes between heavy and light workloads to optimize resource utilization and prevent CUDA out-of-memory errors.

`enable_heavy_gpu_scheduler`: `bool`, default=`False`
: Enable per-GPU heavy task scheduling (default: disabled for backward compatibility)
  - **Heavy models** (VAE, Transformer, CNN): Use per-GPU slot management
  - **Light models** (RandomForest, XGBoost, SVM): Use task-level parallelization
  - **Recommended**: Enable when screening includes deep learning models

`heavy_jobs_per_gpu`: `int`, default=`1`
: Maximum concurrent heavy jobs per GPU device
  - Typical values: 1-2 (depending on GPU memory)
  - Prevents CUDA OOM by limiting concurrent heavy models
  - Example: 4 GPUs × `heavy_jobs_per_gpu=2` = 8 heavy models max

`heavy_max_parallel_jobs`: `Optional[int]`, default=`None`
: Total heavy job cap across all GPUs (None = use GPU slot count)
  - Override GPU slot calculation with a hard limit
  - Example: `heavy_max_parallel_jobs=4` caps at 4 jobs even with 8 GPU slots

`heavy_model_keywords`: `list[str]`, default=`["vae", "transformer", "language_model", "unimol"]`
: Keywords to identify heavy models automatically
  - Models matching these keywords are classified as "heavy"
  - Default covers: VAE, Transformer, language models, UniMol
  - Customize for your model set: `["cnn", "transformer", "vae"]`

**How It Works**:

1. **Automatic Classification**: Models are classified as heavy/light based on `heavy_model_keywords`
2. **Separate Scheduling**:
   - **Heavy tasks**: Queued per-GPU slots (e.g., 2 slots per GPU)
   - **Light tasks**: Task-level parallelization (maximize CPU utilization)
3. **Slot Management**: Each GPU has `heavy_jobs_per_gpu` independent slots
4. **Graceful Degradation**: CUDA errors fall back to CPU on large machines (≥32 cores)

**When to Enable**:
- ✅ **Screening includes VAE/Transformer/CNN** with traditional ML models
- ✅ **Multi-GPU systems** where you want to prevent CUDA OOM
- ✅ **Mixed workloads** where deep learning and sklearn models compete for resources
- ❌ **Traditional ML only** - no benefit, adds unnecessary overhead
- ❌ **Single GPU with 1-2 heavy models** - manual control is simpler

**Example Usage**:

```python
# Mixed workload: VAE + RandomForest + XGBoost
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_heavy_gpu_scheduler=True,      # Enable GPU scheduler
    heavy_jobs_per_gpu=2,                 # 2 heavy jobs per GPU
    heavy_model_keywords=["vae", "transformer", "cnn"],  # Custom heavy keywords
    max_cpu_cores=32                      # Leave room for CPU fallback
)

# Expected behavior with 4 GPUs:
# - VAE models: queued as heavy (max 4×2=8 concurrent)
# - RandomForest/XGBoost: parallelized across CPU cores (light)
# - No CUDA OOM: per-GPU slot enforcement
```

**Performance Impact**:

| Scenario | Without Scheduler | With Scheduler |
|----------|------------------|----------------|
| 4 VAE + 20 RF (4 GPUs) | Risk of CUDA OOM | Stable: 2 VAE per GPU |
| 1 VAE + 30 RF (1 GPU) | VAE blocks all models | VAE takes 1 slot, RF runs on CPU |
| 8 Transformer + 10 XGBoost (2 GPUs) | CUDA errors likely | Controlled: 1 Transformer per GPU |

**CUDA Fallback Behavior**

If a CUDA‑enabled model (VAE/CNN/Transformer/XGBoost) hits a CUDA error during training,
MolBlender will **fall back to CPU only when effective CPU cores ≥ 32**. If fewer
cores are available, the error is raised as usual. This keeps GPU failures from
spamming logs on large CPU machines while preserving strict failure on smaller nodes.

**Progressive Timeout Fallback** ⭐ **NEW**

MolBlender implements a robust timeout control mechanism with progressive fallback. Timeout is automatically managed based on representation type and data size.

**Adaptive Timeout System**:
- **Auto-tuned**: Timeout is automatically adjusted based on representation complexity and dataset size
- **Heavy models**: Longer timeouts for deep learning models (VAE, CNN, Transformer)
- **Light models**: Shorter timeouts for traditional ML (RandomForest, XGBoost, SVM)

**Fallback Chain** (tried in order):
1. **Full Parameters**: `sample_weight + timeout` (try first)
2. **Drop Sample Weight**: If timeout fails with sample_weight, retry without
3. **Drop Timeout**: If still failing, retry with unlimited time
4. **Plain Fit**: Final fallback to basic sklearn fit()

**Timeout Behavior by Model Type**:

| Model Type | Timeout Support | Sample Weight | Behavior |
|------------|---------------|--------------|---------|
| **Deep Learning** (VAE, CNN, Transformer) | ✅ Auto-adaptive | ✅ Via fallback chain | Epoch-level timeout |
| **Traditional ML** (RF, XGBoost, SVM) | ❌ Gracefully ignored | ✅ Full support | Standard sklearn fit |

**Why Progressive Fallback?**
- Deep learning models need timeout to prevent hanging
- Sample weights are important for imbalanced datasets
- Fallback ensures both features work even if one fails
- System automatically handles timeout configuration for all model types

**Log Noise Suppression**

Repeated CUDA/CV/XGBoost failures and near‑constant prediction warnings are deduplicated
to avoid extremely long `screening.out` logs. The first occurrence is kept, subsequent
repeats are suppressed.

**Storage Options**

`enable_db_storage`: `bool`, default=`False`
: Enable SQLite database storage for incremental saving and caching

`db_path`: `str`, optional
: Path to SQLite database file (default: `"screening_results.db"`)

**Hyperparameter Optimization (HPO)**

`enable_hpo`: `bool`, default=`False`
: Enable two-stage hyperparameter optimization for top performers

`hpo_stage`: `str`, default=`"coarse"`
: HPO granularity level
  - `"coarse"` - Fast grid search (3-5 values per parameter)
  - `"fine"` - Detailed grid search (5-10 values per parameter)
  - `"customized"` - User-defined `custom_param_grids`; this stage is
    selected automatically when custom grids are supplied

`hpo_method`: `str`, default=`"grid"`
: Search algorithm for hyperparameter optimization
  - `"grid"` - Exhaustive grid search (recommended)
  - `"random"` - Random search (faster for large grids)

`top_n_for_hpo`: `int`, default=`5`
: Number of top-performing models to optimize in Stage 2

`hpo_cv_folds`: `int`, default=`None`
: Cross-validation folds for HPO (defaults to `cv_folds` if not specified)

**Other Options**

`enable_feature_selection`: `bool`, default=`True`
: Remove zero-variance features (recommended)

`verbose`: `int`, default=`1`
: Logging verbosity
  - `0` = Errors only
  - `1` = Normal progress
  - `2` = Detailed debugging

### Return Value Structure

```python
{
    'success': True,                    # Overall success status
    'best_score': 0.852,                # Best primary metric value
    'best_model': {                     # Best performing model
        'model_name': 'random_forest',
        'representation_name': 'morgan_fp_r2_1024',
        'primary_metric': 0.852,     # Test-set metric value
        'primary_metric_name': 'pearson_r',
        'all_metrics': {
            'r2': 0.852,
            'rmse': 0.543,
            'mae': 0.421,
            'pearson_r': 0.924
        },
        'cv_scores': [0.831, 0.867, 0.849, 0.856, 0.857],
        'cv_mean': 0.852,
        'cv_std': 0.012,
        'training_time': 12.34,
        'model_params': {'n_estimators': 200, 'max_depth': 10},
    },
    'top_models': [...],            # Top results sorted by CV/HPO score
    'all_results': [...],           # All model results sorted by CV/HPO score
    'summary': {                    # Statistical summary
        'n_models_evaluated': 18,
        'n_representations': 6,
        'n_unique_models': 3,
        'best_score': 0.852,
        'mean_score': 0.764,
        'std_score': 0.089
    },
    'screening_config': {...},      # Configuration snapshot
    'timestamp': '2025-01-15T10:30:45'
}
```

### Common Usage Patterns

#### Pattern 1: Default Screening

```python
# Automatic modality detection and model selection
results = universal_screen(
    dataset=dataset,
    target_column="logP"
)
```

#### Pattern 2: Custom Modality Selection

```python
# Screen only fingerprints and descriptors
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    modality_categories=[
        "fingerprints/molecular",
        "descriptors/physicochemical"
    ]
)
```

#### Pattern 3: Comprehensive Screening with Storage

```python
# All models with SQLite storage
results = universal_screen(
    dataset=dataset,
    target_column="bioactivity",
    combinations="comprehensive",
    enable_db_storage=True,
    db_path="./comprehensive_screen.db"
)
```

#### Pattern 4: Memory-Constrained Environment

```python
# Optimize for limited memory
results = universal_screen(
    dataset=large_dataset,
    target_column="solubility",
    execution_preference="memory",
    max_cpu_cores=4,
    cv_folds=3  # Fewer folds = less memory
)
```

#### Pattern 5: Classification Task

```python
# Binary classification with F1 optimization
results = universal_screen(
    dataset=dataset,
    target_column="is_drug_like",
    task_type="classification",
    primary_metric="f1"
)
```

#### Pattern 6: Two-Stage HPO for Top Performers

```python
# Automatic hyperparameter optimization for best models
results = universal_screen(
    dataset=dataset,
    target_column="binding_affinity",
    enable_hpo=True,
    hpo_stage="coarse",     # Fast grid search
    top_n_for_hpo=3,        # Optimize top 3 models
    hpo_cv_folds=3,         # Use 3-fold CV for HPO
    enable_db_storage=True  # Track optimization progress
)
```

### Modality Auto-Detection

MolBlender automatically detects available modalities from your dataset:

```python
# Dataset with SMILES only → AUTO detects STRING modality
dataset = MolecularDataset.from_csv("data.csv", input_column="SMILES")
# Will use: Transformers for raw strings + fingerprints/descriptors (computed on-the-fly)

# Dataset with pre-computed features → AUTO detects VECTOR modality
dataset.add_representation("morgan_fp_r2_1024")
dataset.add_representation("rdkit_descriptors_2d")
# Will use: Traditional ML models (RF, XGBoost, SVM, etc.)

# Dataset with adjacency matrices → AUTO detects MATRIX modality
dataset.add_representation("adjacency_matrix")
# Will use: CNN models + flattened vectors for ML
```

## quick_screen()

Fast screening with essential models for initial exploration.

### Basic Usage

```python
from molblender.models.api import quick_screen

results = quick_screen(
    dataset=dataset,
    target_column="activity"
)
```

### API Reference

```python
def quick_screen(
    dataset,
    target_column: str,
    task_type: str = "regression",       # "regression" or "classification"
    max_cpu_cores: int = -1,             # -1 = all cores
    max_workers_per_model: int = 1,      # per-estimator parallelism
) -> Dict[str, Any]
```

**Key Differences from `universal_screen()`:**
- Tests only 5-10 fast models from the `"fast"` model corpus (RF, XGBoost, Ridge, Bayesian Ridge, KNN)
- Uses 3 CV folds instead of 5 for speed
- No deep learning models (CNN, Transformers)
- No custom representation or test-size configuration — designed for quick exploration
- Optimized for datasets < 10K molecules

### When to Use

✅ Initial data exploration
✅ Baseline performance assessment
✅ Small datasets (<1K molecules)
✅ Time-constrained scenarios
✅ Rapid prototyping

❌ Final model selection
❌ Large datasets
❌ Publication-quality results

## Model Corpus

`screen_models()`, `quick_screen()`, `thorough_screen()`, and `interpretable_screen()` all delegate to `screen_models()` with a preset `model_corpus` value. `ModelCorpus` controls which models are tested:

| **Corpus** | **Contents** | **When to Use** |
|-----------|-------------|----------------|
| `"fast"` | ~5-10 models: RF, XGBoost, Ridge, Bayesian Ridge, KNN | Quick exploration |
| `"essential"` | Fast + a few more standard models | Default quick screen |
| `"accurate"` | Broadest set of traditional ML models | Thorough screening |
| `"interpretable"` | Linear regression, decision tree, sparse linear models | Regulatory submissions, feature importance |

```python
from molblender.models.api import screen_models

results = screen_models(
    dataset=dataset,
    target_column="activity",
    model_corpus="accurate"  # Choose the model set
)
```

## thorough_screen()

Runs `screen_models` with `model_corpus="accurate"`, 5-fold CV, and verbose logging — the most comprehensive standard screening preset without HPO.

```python
from molblender.models.api import thorough_screen

results = thorough_screen(
    dataset=dataset,
    target_column="activity"
)
```

## interpretable_screen()

Runs `screen_models` with `model_corpus="interpretable"`, restricting to linear and tree-based models that expose feature importance.

```python
from molblender.models.api import interpretable_screen

results = interpretable_screen(
    dataset=dataset,
    target_column="activity"
)
```

## screen_models()

The underlying function all preset screeners call. Use it directly when you need a custom `model_corpus` value:

```python
from molblender.models.api import screen_models

results = screen_models(
    dataset=dataset,
    target_column="activity",
    model_corpus="accurate",
    cv_folds=5,
    max_cpu_cores=8,
    max_workers_per_model=2,
)
```

## simple_evaluate()

Test a single model quickly without comprehensive screening.

### Basic Usage

```python
from molblender.models.api import simple_evaluate

result = simple_evaluate(
    dataset=dataset,
    target_column="activity",
    task_type="regression",
    models=["random_forest"],
    representations=["morgan_fp_r2_1024"]
)
```

### API Reference

```python
def simple_evaluate(
    dataset,
    target_column: str,
    representations: List[str] = None,   # Auto-select if None
    models: List[str] = None,             # Auto-select if None
    task_type: str = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
    max_cpu_cores: int = None,
    n_jobs: int = None,
) -> Dict[str, Any]
```

### When to Use

✅ Testing specific model hypothesis
✅ Benchmarking against literature
✅ Quick sanity checks
✅ Debugging

## compare_models()

Compare multiple models on the same representation.

### Basic Usage

```python
from molblender.models.api import compare_models

results = compare_models(
    dataset=dataset,
    target_column="activity",
    representation_name="morgan_fp_r2_1024",
    task_type="regression",
    model_names=["random_forest", "xgboost", "svm_rbf", "ridge"]
)
```

### API Reference

```python
def compare_models(
    dataset,
    target_column: str,
    representation_name: str,
    task_type: str = "regression",
    model_corpus: str = "all",
    model_names: List[str] = None,      # All compatible if None
    statistical_tests: bool = True,     # Significance testing
    cv_folds: int = 5,
    random_state: int = 42,
    max_cpu_cores: int = None,
    n_jobs: int = None,
) -> Dict[str, Any]
```

**Additional Return Values:**
```python
{
    ...
    'statistical_comparison': {
        'friedman_test': {'statistic': 12.34, 'p_value': 0.002},
        'pairwise_comparisons': [
            {'models': ['rf', 'xgb'], 'p_value': 0.045, 'significant': True},
            ...
        ]
    }
}
```

## compare_representations()

Compare multiple representations on the same model.

### Basic Usage

```python
from molblender.models.api import compare_representations

results = compare_representations(
    dataset=dataset,
    target_column="activity",
    model_name="random_forest",
    task_type="regression",
    representation_names=[
        "morgan_fp_r2_1024",
        "morgan_fp_r3_2048",
        "rdkit_descriptors_2d",
        "maccs_keys"
    ]
)
```

### API Reference

```python
def compare_representations(
    dataset,
    target_column: str,
    model_name: str,
    representation_names: List[str],
    task_type: str = "regression",
    statistical_tests: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
    max_cpu_cores: int = None,
    n_jobs: int = None,
) -> Dict[str, Any]
```

## Advanced: Custom Combinations

For fine-grained control, use `Combination` objects:

```python
from molblender.models.api import Combination, universal_screen

custom_combinations = [
    Combination(
        model_name="random_forest",
        representation_name="morgan_fp_r2_1024",
        model_params={'n_estimators': 200, 'max_depth': 10}
    ),
    Combination(
        model_name="xgboost",
        representation_name="rdkit_descriptors_2d",
        model_params={'learning_rate': 0.01, 'n_estimators': 500}
    )
]

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    combinations=custom_combinations
)
```

## Error Handling

All screening functions handle errors gracefully:

```python
results = universal_screen(dataset, target_column="activity")

if not results.get('success', False):
    print(f"Screening failed: {results.get('error')}")
else:
    print(f"Screening succeeded: {results['summary']['n_models_evaluated']} models tested")
```

**Common Errors:**
- `KeyError: target_column` - Target column not found in dataset
- `ValueError: Invalid task_type` - Use "regression" or "classification"
- `MemoryError` - Reduce cv_folds, use execution_preference="memory"
- `TimeoutError` - Individual model timeout (auto-skipped, doesn't halt screening)

## Hyperparameter Optimization (HPO)

MolBlender implements an **intelligent two-stage HPO system** for efficient model tuning.

### Quick Overview

**Stage 1**: Screen all models with default parameters (~10-30 minutes)
**Stage 2**: Optimize top performers with GridSearchCV (~10-60 minutes)

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_hpo=True,           # Enable Stage 2 optimization
    hpo_stage="coarse",        # "coarse" | "fine" | "custom"
    hpo_method="grid",         # "grid" | "random"
    top_n_for_hpo=5,           # Optimize top 5 models
    hpo_selection_scope="global",  # "global" | "per_type" | "per_subtype"
    hpo_cv_folds=3,            # Use 3-fold CV for HPO (faster)
    enable_db_storage=True     # Track all stages in database
)
```

### HPO Selection Strategies

`hpo_selection_scope` controls which models are selected for Stage 2 optimization:

- **`"global"`** (default) - Top N models overall by primary metric
- **`"per_type"`** - Top N Traditional ML + Top N Deep Learning
- **`"per_subtype"`** - Top N from each model family (LINEAR, TREE, BOOSTING, KERNEL, VAE, TRANSFORMER, CNN)

```{admonition} For Complete HPO Documentation
:class: seealso

See {doc}`hpo` for comprehensive HPO guide including:
- Selection strategies and model subtypes
- Parameter grid configurations
- Custom grid definitions
- Database integration and resuming
- Performance tips and troubleshooting
```

## Performance Tips

```{admonition} Best Practices
:class: tip

1. **Start with `universal_screen()` default settings** - balanced performance
2. **Enable database storage** for runs > 10 minutes - `enable_db_storage=True`
3. **Use 3 CV folds for large datasets** (>10K molecules) - faster with minimal accuracy loss
4. **Monitor with `verbose=2`** when debugging - detailed progress logging
5. **Leave 2 cores free** for system - `max_cpu_cores=-2`
6. **Enable HPO for final models** - `enable_hpo=True` after initial screening
```

## Next Steps

- **See available models**: {doc}`models` - Complete model catalog
- **Access results**: {doc}`results` - SQLite database and exports
- **Visualize performance**: {doc}`../dashboard/index` - Interactive dashboard
