# Quick Start

A comprehensive end-to-end example demonstrating how to use MolBlender for molecular property prediction, from data loading through result analysis.

## Overview

This tutorial walks through a complete molecular machine learning workflow:

1. **Data Loading & Validation** - Import molecular datasets from various formats
2. **Representation Generation** - Create diverse molecular features
3. **Automated Model Screening** - Find optimal model+representation combinations
4. **Results Visualization** - Generate publication-quality plots
5. **Results Persistence** - Save, reload, and reuse screening results

## Choosing the Right API

MolBlender provides multiple API layers. For this tutorial, we use:

```python
# Recommended: Unified API (molblender.api)
from molblender.api import get_featurizer, run_dashboard, load_results, list_featurizers

# Screening entry points (canonical home after the 2026-09 package promotion)
from molblender.screening import universal_screen, thorough_screen, quick_screen

# Detailed featurizer metadata
from molblender.representations import get_featurizer_info

# Static plots
from molblender.drawings import plot_model_comparison
```

**Quick guide**:
- **New code**: Use `molblender.api` for common tasks, `molblender.screening` for full screening control
- **Domain APIs**: `molblender.representations`, `molblender.data`, `molblender.drawings`
- **Interactive exploration**: `molblender.api.run_dashboard(...)` or `molblender view`
- **Old `molblender.models.api.*` paths** still import through deprecation adapters (removed in v2.0) — do not use them in new code

Notes:
- `import molblender` and `import molblender.api` are both lightweight lazy facades.
- `molblender.data` is also a lazy facade over dataset, diagnostics, cache, and preprocessing subdomains.
- The screening implementation lives in `molblender.screening` (`engine`, `orchestration`, `runtime`); most users only need the public functions re-exported at `molblender.screening` level.

See [API Guide](api_guide.md) for detailed API layer documentation.

## Dataset: Predicting Molecular Solubility

We'll predict aqueous solubility using a small example set of drug-like molecules.

### Step 1: Data Loading and Exploration

```python
import pandas as pd
import numpy as np
from molblender.data import MolecularDataset
from molblender.screening import thorough_screen
from molblender.drawings import plot_model_comparison

# Load solubility dataset (example data)
# In practice, download from: https://www.moleculenet.org/datasets-1
solubility_data = pd.DataFrame({
    'SMILES': [
        'CCO',                                    # Ethanol
        'c1ccccc1O',                             # Phenol
        'CC(=O)OC1=CC=CC=C1C(=O)O',             # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',        # Caffeine
        'CC1=CC=C(C=C1)C(=O)O',                 # p-Toluic acid
        'CCCCCCCCCC(=O)O',                       # Decanoic acid
        'c1ccc2c(c1)ccc3c2ccc4c3cccc4',         # Anthracene
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'        # Ibuprofen
    ],
    'solubility': [-0.77, -0.04, -2.23, -0.07, -2.25, -4.09, -7.56, -3.97]  # LogS values
})

# Create MolBlender dataset
dataset = MolecularDataset.from_dataframe(
    solubility_data,
    input_column='SMILES',
    label_columns=['solubility']
)

print(f"Dataset loaded: {len(dataset)} molecules")
print(f"Solubility range: {dataset.labels['solubility'].min():.2f} to {dataset.labels['solubility'].max():.2f}")

# Data validation
validation_report = dataset.validate()
print(f"All molecules valid: {validation_report['all_valid']}")
```

For custom loaders and preprocessing code, reuse low-level shared checks from
`molblender.validation` (for example file existence, suffix checks, numeric
ranges, and required DataFrame columns) instead of reimplementing them in each
module.

### Step 2: Generate Diverse Molecular Representations

```python
# Select diverse representation types for comprehensive screening
representations = [
    # Fingerprints - structural patterns
    "morgan_fp_r2_1024",           # Circular fingerprints
    "rdkit_fp_1024",               # RDKit path-based fingerprints
    "maccs_keys",                  # 166-bit pharmacophore keys

    # Descriptors - physicochemical properties
    "rdkit_all_descriptors",       # RDKit descriptor block
    "mordred_descriptors_2d",      # Mordred 2D descriptors
]

# Add representations with parallel processing
print("Computing molecular representations...")
for repr_name in representations:
    try:
        dataset.add_features(repr_name, n_workers=4)
        n_failed = len(dataset.get_featurization_failures(repr_name))
        status = "OK" if n_failed == 0 else f"{n_failed} failures"
        print(f"  {repr_name}: {status}")
    except Exception as e:
        print(f"  {repr_name}: FAILED - {e}")

# Show which representations are now attached
print(f"\nFeatures computed: {dataset.feature_names}")
```

`add_features` returns `None`; use `get_featurization_failures(name)` to inspect
per-molecule failures and `drop_featurization_failures(name)` to remove failed
rows. Not every representation fits every dataset — 3D representations such as
`coulomb_matrix` require conformers and may fail on SMILES-only input.

### Step 3: Automated Model Screening

```python
# Thorough screening: accurate model corpus, 5-fold CV
print("Starting automated model screening...")

screening_results = thorough_screen(
    dataset=dataset,
    target_column='solubility',
    task_type='regression',
    max_cpu_cores=4,          # Total CPU budget for screening
    max_workers_per_model=1,  # Per-model worker cap
)

# Display results
best = screening_results['best_model']
print(f"\nScreening Results:")
print(f"Success: {screening_results['success']}")
print(f"Best Model: {best['model_name']} + {best['representation_name']}")
print(f"Performance: {best['primary_metric_name']} = {best['primary_metric']:.3f} "
      f"(CV {best['cv_mean']:.3f} ± {best['cv_std']:.3f})")

# Show top models
print(f"\nTop Models:")
for model_info in screening_results['top_models'][:3]:
    print(f"  #{model_info['rank']}: {model_info['model_name']} + "
          f"{model_info['representation_name']} "
          f"({model_info['primary_metric_name']} = {model_info['primary_metric']:.3f})")
```

The result dictionary contains `best_model`, `top_models`, `all_results`,
`summary`, and `performance_analysis` keys. Each model entry carries
`model_name`, `representation_name`, `primary_metric`, `all_metrics`,
`cv_mean`/`cv_std`, and optional `predictions`.

For full control over the modality/representation sweep (including
representations that are not pre-computed on the dataset), use
`universal_screen` with grouped configs:

```python
from molblender.screening import universal_screen
from molblender.screening.orchestration import SplitConfig

results = universal_screen(
    dataset=dataset,
    target_column='solubility',
    modality_categories=['fingerprints', 'descriptors'],
    combinations='auto',
    split_config=SplitConfig(test_size=0.25, cv_folds=3),
)
```

### Step 4: Results Visualization

```python
# Compare the top models by the screening primary metric
fig, ax = plot_model_comparison(
    screening_results['top_models'],
    metric='primary_metric',
    ylabel=best['primary_metric_name'],
    title="Model Performance Comparison",
)
fig.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
print("Comparison plot saved: model_comparison.png")
```

For interactive exploration (sortable tables, per-model inspection, dynamic
metric switching), launch the dashboard:

```python
from molblender.api import run_dashboard
run_dashboard(results_path="screening_results.db")
```

or from the command line: `molblender view screening_results.db`. See the
[Dashboard guide](usage/dashboard/index.md) for all tabs and export options.

### Step 5: Results Persistence and Reuse

```python
# Persist results through screen_models-compatible parameters
from molblender.screening import screen_models

results = screen_models(
    dataset=dataset,
    target_column='solubility',
    save_results=True,
    output_path="screening_results.json",
    enable_db_storage=True,
    db_path="screening_results.db",
)

# Reload stored results later (from the SQLite database)
from molblender.api import load_results
stored = load_results("screening_results.db")
print(f"Stored results for {len(stored.get('all_results', []))} combinations")
```

To deploy the best combination, retrain it explicitly — the screening result
records what won, and the featurizer + estimator pair is rebuilt by name:

```python
from molblender import get_featurizer
from sklearn.ensemble import RandomForestRegressor

best_repr = best['representation_name']
featurizer = get_featurizer(best_repr)
X = dataset.get_features(best_repr)
y = dataset.get_labels('solubility')

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

def predict_solubility(smiles_list):
    X_new = featurizer.featurize(smiles_list)
    return model.predict(X_new)

print(predict_solubility(['CCO', 'CCCO', 'CCCCO']))  # Alcohols of increasing chain length
```

## Summary and Best Practices

This workflow demonstrates MolBlender's key capabilities:

```{admonition} Key Takeaways
:class: tip

**Data Management:**
- Always validate your dataset with `dataset.validate()`
- Handle invalid molecules gracefully with `add_features(..., on_error=...)`
  and `get_featurization_failures()`
- Use multiple file format support for flexible data loading

**Representation Selection:**
- Start with diverse representation types (fingerprints, descriptors, 3D)
- Use `thorough_screen()` for the accurate corpus or `universal_screen()` for
  full modality sweeps
- Consider computational cost vs. accuracy trade-offs

**Model Screening:**
- Use `max_cpu_cores` for total screening parallelism and
  `max_workers_per_model` for model-internal parallelism
- Enable `enable_db_storage` so results can be explored in the Dashboard and
  merged across sessions
- Read the best entry from `results['best_model']`, not from ad-hoc keys

**Results Analysis:**
- Generate publication-quality visualizations with `molblender.drawings`
- Explore interactively with `molblender view`
- Validate performance with proper cross-validation

**Deployment:**
- Retrain the winning featurizer + model pair explicitly for production
- Save screening results (JSON/SQLite) for reproducibility
- Document model performance and limitations
```

### Scaling to Larger Datasets

For datasets with >10,000 molecules:

```python
from molblender.screening import quick_screen, thorough_screen

# Fast first pass
quick_results = quick_screen(
    large_dataset,
    target_column="target",
    max_cpu_cores=8,
)

# Then detailed screening on promising representations
detailed_results = thorough_screen(
    large_dataset,
    target_column="target",
    max_cpu_cores=16,
)
```

To restrict the representation sweep, pass `representation_names=[...]` to
`screen_models`, or `modality_categories`/`combinations` to `universal_screen`.

### Next Steps

- Explore specialized representations for your domain
- Implement custom featurizers for novel descriptors
- Use the visualization module for advanced plotting
- Deploy models in production environments
- Extend to multi-task and multi-modal learning

This complete workflow showcases MolBlender's power in automating molecular machine learning while maintaining flexibility and interpretability.

### Advanced: Featurizer Catalog and Query APIs

For advanced featurizer discovery and metadata queries, use the catalog and query
helpers exposed from `molblender.representations`:

```python
from molblender.representations import FeaturizerCatalog, FeaturizerQuery

# List all available featurizers
all_featurizers = FeaturizerCatalog.list_all(include_protein=True)
print(f"Total featurizers: {len(all_featurizers)}")

# Get fingerprint featurizers
fingerprints = FeaturizerQuery.by_category("fingerprints")
print(f"Fingerprint featurizers: {[f.name for f in fingerprints]}")

# Filter by tag
gpu_featurizers = FeaturizerQuery.by_tag("gpu")
print(f"GPU featurizers: {[f.name for f in gpu_featurizers]}")

# Get detailed metadata
info = FeaturizerCatalog.get_info("morgan_fp_r2_1024")
if info:
    print(f"Name: {info.name}")
    print(f"Category: {info.category}")
    print(f"Description: {info.description}")
    print(f"Source: {info.source}")
    print(f"Output shape: {info.output_shape}")

# Search featurizers
results = FeaturizerQuery.search("fingerprint")
for info in results:
    print(f"{info.name}: {info.description}")
```
