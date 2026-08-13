# Dataset Diagnostics

Diagnose dataset quality before modeling — identify rough SAR regions, activity cliffs, and diversity gaps.

## Overview

`DatasetDiagnostics` provides a one-click quality report for a molecular dataset. It computes DNR (Different Neighbor Ratio), detects activity cliff pairs, and generates publication-quality visualizations.

```{admonition} Use Case
:class: note

Before investing time in model training, run diagnostics to understand:
- Which molecules sit in **rough SAR regions** where similar structures have very different activities (high DNR)
- Whether **activity cliffs** exist — pairs of near-identical molecules with large property jumps
- How **diverse** the dataset is and whether certain regions are over- or under-represented

High DNR regions predictably hurt model performance; removing or stratifying them can improve results.
```

## Quick Start

```python
from molblender.data import MolecularDataset, InputType
from molblender.data.diagnostics import DatasetDiagnostics

dataset = MolecularDataset.from_csv(
    "data.csv",
    input_column="SMILES",
    mol_input_type=InputType.SMILES,
    label_columns=["pIC50"]
)

diagnostics = DatasetDiagnostics(dataset)
report = diagnostics.run_full_diagnostics(
    label_col="pIC50",
    save_dir="./diagnostics/"
)

print(f"High DNR molecules: {report['dnr_analysis']['n_high_dnr']}")
print(f"Activity cliff pairs: {report['activity_cliffs']['n_cliffs']}")
```

## DNR (Different Neighbor Ratio)

DNR measures local SAR smoothness for each molecule, based on:

> "Upgrading Reliability in Molecular Property Prediction by Robust Quantification of Uncertainty from Machine Learning"

**Definition:**

$$
\text{DNR}(x, y) = \frac{N_{\text{different}}(x, y)}{N_{\text{total}}(x)}
$$

- $N_{\text{total}}(x)$: number of neighbors (Tanimoto similarity > 0.5)
- $N_{\text{different}}(x, y)$: neighbors with $|y_i - y_j| \geq 1.0$ log unit difference
- If a molecule has **no neighbors**, DNR = 0

**Interpretation:**

| DNR Range | Meaning | Implication |
|-----------|---------|-------------|
| DNR = 0 | No neighbors or all neighbors share similar activity | Model can learn smoothly |
| DNR < 0.2 | Smooth local SAR | Low modeling difficulty |
| DNR > 0.5 | Rough SAR, activity cliffs present | High modeling difficulty |

### Computing DNR

```python
dnr_df = diagnostics.calculate_dnr(
    label_col="pIC50",
    similarity_threshold=0.5,       # Tanimoto threshold
    property_diff_threshold=1.0,    # log unit difference
    fingerprint_radius=2,
    fingerprint_bits=2048
)

# Filter high-DNR molecules
high_dnr = dnr_df[dnr_df['is_high_dnr']]
print(f"{len(high_dnr)} molecules with DNR > 0.5")

# Filter isolated molecules (no neighbors)
isolated = dnr_df[~dnr_df['has_neighbors']]
print(f"{len(isolated)} isolated molecules")
```

The returned DataFrame has one row per molecule in the original dataset, indexed by integer position. Columns:

| Column | Description |
|--------|-------------|
| `mol_id` | Molecule ID from `dataset.ids` |
| `n_total_neighbors` | Total neighbors (Tanimoto > threshold) |
| `n_different_neighbors` | Neighbors with property difference above threshold |
| `dnr` | DNR score [0, 1] |
| `is_high_dnr` | Boolean, DNR > 0.5 |
| `has_neighbors` | Whether the molecule has any neighbors |
| `property_value` | Property value for this molecule |
| `is_invalid` | True if SMILES could not be parsed |
| `out_of_sample` | True if excluded by auto-sampling (datasets > 10,000 molecules) |

## Activity Cliffs

An **activity cliff** is a pair of molecules that are structurally similar but have very different activities — the hardest cases for ML models.

```python
cliffs_df = diagnostics.detect_activity_cliffs(
    label_col="pIC50",
    similarity_threshold=0.5,
    property_diff_threshold=1.0
)

print(f"Found {len(cliffs_df)} activity cliff pairs")
print(cliffs_df.head())
```

The returned DataFrame contains:

| Column | Description |
|--------|-------------|
| `mol_id_1`, `mol_id_2` | Molecule IDs of the cliff pair |
| `similarity` | Tanimoto similarity between the pair |
| `property_1`, `property_2` | Activity values for each molecule |
| `property_diff` | Absolute difference between the two |

## Full Diagnostic Report

`run_full_diagnostics` runs the complete pipeline in one call:

```python
report = diagnostics.run_full_diagnostics(
    label_col="pIC50",
    save_dir="./diagnostics/"
)
```

The report dictionary contains:

| Key | Contents |
|-----|----------|
| `basic_statistics` | `n_molecules`, `n_valid`, `label_statistics`, `molecular_weight_stats` |
| `dnr_analysis` | `n_high_dnr`, `high_dnr_rate`, `n_no_neighbors`, `no_neighbor_rate`, `mean_dnr`, `std_dnr`, `dnr_stats` |
| `activity_cliffs` | `n_cliffs`, `mean_similarity`, `mean_diff`, `max_diff` |
| `dnr_dataframe` | Full DNR DataFrame (one row per molecule) |
| `cliff_dataframe` | Full activity cliff pairs DataFrame |

When `save_dir` is provided, four SVG visualizations are generated:

| File | Description |
|------|-------------|
| `dnr_distribution.svg` | Histogram of DNR values across the dataset |
| `activity_cliffs.svg` | Network graph of cliff pairs |
| `dnr_vs_property.svg` | Scatter plot of DNR vs. activity value |
| `similarity_heatmap.svg` | Tanimoto similarity heatmap (only for datasets ≤ 100 molecules) |

## Similarity Utilities

Standalone functions are available for custom analysis:

```python
from molblender.data.diagnostics import (
    compute_morgan_fingerprints,
    compute_tanimoto_similarity_matrix,
    compute_tanimoto_similarity_result,
    find_neighbors,
)

fps = compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048)
sim_matrix = compute_tanimoto_similarity_matrix(fps)
neighbors = find_neighbors(sim_matrix, threshold=0.5)
```

## Workflow Recommendations

```{admonition} Practical Guide
:class: tip

- **Before screening**: Run `run_full_diagnostics` to understand modeling difficulty
- **High DNR rate > 30%**: Consider removing high-DNR molecules or using scaffold-based splitting
- **Many activity cliffs**: Models will struggle — try uncertainty-aware approaches
- **Many isolated molecules (no neighbors)**: These contribute little to training; consider whether they represent true chemical diversity or data collection gaps
```

## See Also

- **Data splitting**: {doc}`splitting/index` — choose splits that respect SAR structure
- **Model screening**: {doc}`../../models/screening` — run benchmarks after diagnostics