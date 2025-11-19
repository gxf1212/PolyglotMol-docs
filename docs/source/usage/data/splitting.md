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

PolyglotMol supports 10 different splitting strategies, each suited for specific scenarios:

| Strategy | Use Case | Train/Val/Test | Best For |
|----------|----------|----------------|----------|
| `train_test` | Standard screening | 80% / — / 20% | Most common scenarios |
| `train_val_test` | With hyperparameter optimization | 70% / 15% / 15% | Large datasets with HPO |
| `nested_cv` | Unbiased HPO performance | Nested CV | Academic research |
| `cv_only` | Small datasets | CV only | < 100 samples |
| `scaffold` | Drug discovery | Scaffold-based | Novel structure generalization |
| `dnr` | Rough SAR analysis | DNR-based | Testing on challenging molecules |
| `maxmin` | Diversity testing | MaxMin-based | Chemical space extrapolation |
| `butina` | Cluster-based validation | Butina clustering | Similar molecule generalization |
| `feature_clustering` | General clustering split | K-means/Hierarchical/DBSCAN | Custom representations (3D, embeddings) |
| `user_provided` | Custom splits | User-defined | Temporal, custom logic |

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

### 5. Scaffold Split

Scaffold-based splitting for drug discovery and medicinal chemistry applications.

#### Configuration

```python
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="scaffold",
    test_size=0.2,
    scaffold_func='bemis_murcko',  # or 'generic'
    split_method='balanced',       # or 'random'
    random_state=42
)
```

#### How It Works

Scaffold split ensures that molecules in the test set have **different chemical scaffolds** from those in the training set, providing a more realistic evaluation of model generalization to novel structures.

```
Dataset (1000 molecules)
    ↓
Compute Bemis-Murcko scaffolds
    ↓
Group molecules by scaffold
    Scaffold A: 400 molecules (benzene derivatives)
    Scaffold B: 350 molecules (pyridine derivatives)
    Scaffold C: 150 molecules (furan derivatives)
    Scaffold D: 100 molecules (thiophene derivatives)
    ↓
Assign entire scaffold groups to train or test
    ├─→ Training Set: Scaffolds A, B (750 molecules, 75%)
    └─→ Test Set: Scaffolds C, D (250 molecules, 25%)
           ↓
        ✓ No scaffold leakage between train and test
```

#### Scaffold Generation Methods

**1. Bemis-Murcko Scaffolds (default)**

Extracts the core ring system and linker atoms, removing all side chains.

```python
from polyglotmol.models.api.core.splitting import compute_bemis_murcko_scaffolds

smiles = [
    "c1ccccc1CCO",      # Benzene + ethanol side chain
    "c1ccccc1CCN",      # Benzene + ethylamine side chain
    "c1ccccc1C(=O)O",   # Benzene + carboxylic acid
]

scaffolds = compute_bemis_murcko_scaffolds(smiles)
# All three have the same scaffold: "c1ccccc1" (benzene)
```

**2. Generic Scaffolds**

Further abstracts by replacing all atoms with carbons and all bonds with single bonds, focusing purely on topology.

```python
from polyglotmol.models.api.core.splitting import compute_generic_scaffolds

smiles = [
    "c1ccccc1CCO",    # Aromatic benzene ring
    "C1CCCCC1CCO",    # Aliphatic cyclohexane ring
]

generic_scaffolds = compute_generic_scaffolds(smiles)
# Both have the same generic scaffold: "C1CCCCC1" (6-membered ring)
```

#### Split Methods

**1. Balanced Split (default)**

Greedily assigns scaffold groups to achieve the desired train/test ratio.

```python
# Balanced method - aims for exact test_size
result = scaffold_split(
    X, y, smiles,
    split_method='balanced',  # Balances dataset sizes
    test_size=0.2
)
```

**Strategy:**
1. Sort scaffold groups by size (largest first)
2. Greedily assign to train/test to match target ratio
3. Result: test set close to desired size

**2. Random Split**

Randomly assigns each scaffold group to either train or test.

```python
# Random method - more stochastic
result = scaffold_split(
    X, y, smiles,
    split_method='random',  # Random scaffold assignment
    test_size=0.2
)
```

**Strategy:**
1. Randomly shuffle unique scaffolds
2. Assign scaffolds to train/test until target ratio met
3. Result: more variability across runs (with different seeds)

#### Use Cases

**Drug Discovery:**
- **Generalization to novel scaffolds**: Test model's ability to predict activity for chemically distinct structures
- **Lead optimization**: Evaluate performance on scaffold hops
- **Virtual screening**: More realistic estimate of hit rate on diverse libraries

**Example:**
```python
from polyglotmol.data import MolecularDataset
from polyglotmol.models import universal_screen

# Load drug-like molecules
dataset = MolecularDataset.from_csv(
    "kinase_inhibitors.csv",
    input_column="SMILES",
    label_columns=["pIC50"]
)

# Scaffold-based screening
results = universal_screen(
    dataset=dataset,
    target_column="pIC50",
    split_strategy="scaffold",      # Use scaffold split
    test_size=0.2,
    scaffold_func='bemis_murcko',   # Standard BM scaffolds
    split_method='balanced',
    random_state=42
)

# Results reflect generalization to novel scaffolds
print(f"Test R²: {results['best_model']['test_r2']:.3f}")
```

#### Direct Usage

For more control, use the scaffold split function directly:

```python
from polyglotmol.models.api.core.splitting import scaffold_split
import numpy as np

# Your data
smiles = ["c1ccccc1CCO", "c1ccccc1CCN", ...]  # List of SMILES
X = np.array([...])  # Feature matrix
y = np.array([...])  # Target values

# Perform scaffold split
result = scaffold_split(
    X, y, smiles,
    test_size=0.2,
    scaffold_func='bemis_murcko',
    split_method='balanced',
    random_state=42
)

# Access results
X_train = result['X_train']
X_test = result['X_test']
y_train = result['y_train']
y_test = result['y_test']
scaffolds = result['scaffolds']  # Scaffold for each molecule

# Verify no scaffold leakage
train_scaffolds = set(scaffolds[i] for i in result['train_indices'])
test_scaffolds = set(scaffolds[i] for i in result['test_indices'])
overlap = train_scaffolds & test_scaffolds
print(f"Scaffold overlap: {len(overlap)} scaffolds")  # Should be 0
```

#### Implementation Details

**Code Location**: `src/polyglotmol/models/api/core/splitting/scaffold.py`

```python
def scaffold_split(
    X: np.ndarray,
    y: np.ndarray,
    smiles: List[str],
    test_size: float = 0.2,
    scaffold_func: str = 'bemis_murcko',
    split_method: str = 'balanced',
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform scaffold-based train/test split.

    Returns:
        Dictionary with keys: X_train, X_test, y_train, y_test,
        train_indices, test_indices, scaffolds, split_type
    """
```

#### Advantages

```{admonition} Why Use Scaffold Split?
:class: tip

**1. Realistic Evaluation**: Mimics real-world scenarios where you predict activity for novel chemical matter

**2. Prevents Data Leakage**: Ensures no similar structures in both train and test sets

**3. Conservative Estimates**: Usually gives lower performance than random split, providing a more honest assessment

**4. Industry Standard**: Widely used in pharmaceutical industry for virtual screening validation
```

#### Comparison with Random Split

```python
# Random split (optimistic)
random_results = universal_screen(
    dataset, "pIC50",
    split_strategy="train_test",  # Random split
    test_size=0.2,
    random_state=42
)
print(f"Random split R²: {random_results['best_model']['test_r2']:.3f}")

# Scaffold split (realistic)
scaffold_results = universal_screen(
    dataset, "pIC50",
    split_strategy="scaffold",  # Scaffold split
    test_size=0.2,
    random_state=42
)
print(f"Scaffold split R²: {scaffold_results['best_model']['test_r2']:.3f}")

# Typical result: Scaffold split R² < Random split R²
# Example: 0.65 vs 0.78
```

```{note}
Scaffold split typically yields **lower performance metrics** than random split because it tests generalization to truly novel structures. This is expected and provides a more realistic estimate of real-world performance.
```

### 6. Butina Clustering Split

Leave-cluster-out validation based on Tanimoto similarity clustering, preventing information leakage from chemically similar molecules.

#### What is Butina Clustering?

Butina clustering is a sphere exclusion algorithm that groups molecules based on structural similarity:
- **Automatic clustering**: Self-adaptive cluster count based on similarity threshold (no need to specify K)
- **Leave-cluster-out**: Entire clusters move as units to train or test set
- **Similarity-based**: Uses Tanimoto similarity on molecular fingerprints

```{admonition} Use Case
:class: tip

Butina split is ideal when you want to:
- Test generalization to **similar but unseen** chemical combinations
- Avoid information leakage from structural similarity
- Implement MolAgent-style clustering validation
- Evaluate on OP-like structured design spaces (R group + Y functional group combinations)
```

#### Configuration

```python
from polyglotmol.models import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="dG",
    split_strategy="butina",           # Butina clustering
    test_size=0.2,
    butina_similarity_threshold=0.6,   # Tanimoto threshold
    fingerprint_type='morgan',         # Morgan/RDKit/MACCS
    fp_radius=2,
    fp_nbits=2048,
    random_state=42
)

# Check cluster statistics
print(f"Number of clusters: {results['split_info']['n_clusters']}")
print(f"Actual test size: {results['split_info']['test_size_actual']:.1%}")
print(f"Train intra-similarity: {results['split_info']['train_avg_intra_similarity']:.3f}")
print(f"Test intra-similarity: {results['split_info']['test_avg_intra_similarity']:.3f}")
print(f"Train-test similarity: {results['split_info']['train_test_similarity']:.3f}")
```

#### How It Works

```
Dataset (120 molecules)
    ↓
Compute Tanimoto similarity matrix (Morgan fingerprints)
    ↓
Butina clustering (similarity_threshold=0.6)
    ├─→ Cluster 1: 15 molecules (all Tanimoto > 0.6)
    ├─→ Cluster 2: 12 molecules
    ├─→ Cluster 3: 8 molecules
    ├─→ ...
    └─→ Cluster N: 5 molecules
       ↓
Greedy balanced assignment (largest clusters first)
    ├─→ Test set: Clusters 1, 3, 5 (24 molecules, 20%)
    └─→ Train set: Clusters 2, 4, 6, ... (96 molecules, 80%)
       ↓
Leave-cluster-out validation
(No molecules from same cluster split across train/test)
```

#### Direct Usage

```python
from polyglotmol.models.api.core.splitting import butina_split

# Perform Butina clustering split
result = butina_split(
    dataset=dataset,
    test_size=0.2,
    similarity_threshold=0.6,  # Molecules with Tanimoto > 0.6 cluster together
    fingerprint_type='morgan',
    radius=2,
    nbits=2048,
    random_state=42
)

# Access results
train_idx = result['train_indices']
test_idx = result['test_indices']
cluster_info = result['split_info']

print(f"Clusters: {cluster_info['n_clusters']}")
print(f"Largest cluster: {cluster_info['largest_cluster_size']} molecules")
print(f"Smallest cluster: {cluster_info['smallest_cluster_size']} molecules")
```

#### Implementation Details

**Code Location**: `src/polyglotmol/models/api/core/splitting/butina.py`

```python
def butina_split(
    dataset: MolecularDataset,
    test_size: float = 0.2,
    similarity_threshold: float = 0.6,
    fingerprint_type: str = 'morgan',
    radius: int = 2,
    nbits: int = 2048,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Butina clustering + leave-cluster-out splitting.

    Uses Butina's sphere exclusion algorithm to cluster molecules,
    then assigns entire clusters to train or test sets.
    """
```

**Algorithm**: Butina, D. J. Chem. Inf. Comput. Sci. 1999, 39, 747-750

#### Butina vs Scaffold Split

| Feature | Butina | Scaffold |
|---------|--------|----------|
| **Granularity** | Fine-grained (full molecule topology) | Coarse-grained (core structure only) |
| **Considers** | All atoms + substituents + functional groups | Only ring systems + linkers |
| **Use Case** | Similar molecule generalization | Novel scaffold generalization |
| **Example** | Benzene ↔ Toluene (similar) | Benzene ↔ Pyridine (different scaffolds) |

```python
# Example: OP-based dataset (R groups + Y functional groups)
# Scaffold: Only considers backbone structure (too coarse)
# Butina: Considers R + Y combinations (appropriate granularity)

# Molecules with same scaffold but different substituents
mol1 = "Polymer-N+(cyclohexyl)-O-...-CO2-"  # R=cyclohexyl, Y=carboxyl
mol2 = "Polymer-N+(cycloheptyl)-O-...-SO3-"  # R=cycloheptyl, Y=sulfonate

# Scaffold split: Same scaffold → likely both in train or both in test
# Butina split: Different Tanimoto → may be in different clusters
```

#### Advantages

```{admonition} Why Use Butina Split?
:class: tip

**1. Prevents Similarity Bias**: Avoids over-optimistic evaluation from similar molecules in train/test

**2. Self-Adaptive**: Automatic cluster count based on similarity threshold

**3. Cluster Integrity**: Entire clusters move as units (true leave-cluster-out)

**4. Fine-Grained Control**: More granular than scaffold (considers full molecular structure)

**5. MolAgent Alignment**: Implements cluster-based validation from MolAgent literature
```

### 7. Feature Clustering Split

General-purpose clustering split supporting arbitrary molecular representations (not limited to fingerprints).

#### What is Feature Clustering?

Feature clustering split uses general machine learning clustering algorithms (K-means, Hierarchical, DBSCAN) to partition molecules based on **user-defined feature spaces**, then performs leave-cluster-out splitting:

- **Flexible Feature Sources**: User representations, RDKit descriptors, or fingerprints
- **Multiple Algorithms**: K-means (fast, spherical), Hierarchical (tree-based), DBSCAN (density-based)
- **Auto K-Selection**: Silhouette score optimization for optimal cluster count
- **Quality Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin indices

Feature clustering split is ideal when:

- **Non-Fingerprint Representations**: Using 3D embeddings (Boltz-2), language models (ChemBERTa), quantum features
- **Custom Feature Spaces**: Physicochemical descriptors, docking scores, or domain-specific features
- **Flexible Clustering**: Need control over clustering algorithm (K-means vs DBSCAN)
- **Representation Evaluation**: Testing model generalization across different representation space regions

#### Configuration

```python
from polyglotmol.models import universal_screen
from polyglotmol.data import MolecularDataset

# Using RDKit descriptors with K-means
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    split_strategy="feature_clustering",

    # Clustering parameters
    clustering_algorithm="kmeans",   # 'kmeans', 'hierarchical', 'dbscan'
    n_clusters=5,                    # Manual k (None for auto-selection)
    auto_select_k=False,             # Set True to auto-optimize k

    # Feature source (priority: features > descriptors > fingerprints)
    use_descriptors=True,            # Use RDKit descriptors (~20 features)
    fingerprint_type="morgan",       # Fallback if use_descriptors=False

    # DBSCAN-specific (only if algorithm='dbscan')
    dbscan_eps=0.5,                  # Max distance between neighbors
    dbscan_min_samples=5,            # Min cluster size

    test_size=0.2,
    random_state=42
)
```

#### How It Works

```
1. Feature Extraction
   ├─ User-provided features (if provided) → Use directly
   ├─ RDKit descriptors (if use_descriptors=True) → Compute MW, LogP, TPSA, etc.
   └─ Fingerprints (fallback) → Morgan/RDKit/MACCS

2. Feature Standardization
   └─ StandardScaler normalization (zero mean, unit variance)

3. Clustering
   ├─ K-means: Spherical clusters, need k specification
   ├─ Hierarchical: Ward linkage, tree-based structure
   └─ DBSCAN: Density-based, auto cluster count

4. Optimal K Selection (if auto_select_k=True)
   └─ Test k ∈ [2, min(10, √n)]
   └─ Select k maximizing Silhouette score

5. Leave-Cluster-Out Assignment
   ├─ Sort clusters by size (descending)
   ├─ Greedy assignment: Largest clusters → test set first
   ├─ Target test_size proportion
   └─ DBSCAN: Noise points (-1 label) → train set

6. Quality Metrics
   ├─ Silhouette score: [-1, 1], >0.5 = good clustering
   ├─ Calinski-Harabasz: Higher = better-defined clusters
   └─ Davies-Bouldin: Lower = better separation
```

#### Direct Usage

```python
from polyglotmol.models.api.core.splitting import feature_clustering_split
from polyglotmol.data import MolecularDataset

# Example 1: User-provided features (3D embeddings)
from polyglotmol.representations import SomeEmbeddingGenerator

generator = SomeEmbeddingGenerator()
features = generator.generate(dataset)  # Shape: (n_molecules, n_features)

splits = feature_clustering_split(
    dataset=dataset,
    features=features,              # Provide custom features
    clustering_algorithm='kmeans',
    n_clusters=None,                # Auto-select optimal k
    auto_select_k=True,
    test_size=0.2,
    random_state=42
)

# Example 2: RDKit descriptors with hierarchical clustering
splits = feature_clustering_split(
    dataset=dataset,
    use_descriptors=True,           # Use ~20 RDKit descriptors
    clustering_algorithm='hierarchical',
    auto_select_k=True,
    test_size=0.25,
    random_state=42
)

# Example 3: DBSCAN with fingerprints
splits = feature_clustering_split(
    dataset=dataset,
    clustering_algorithm='dbscan',
    eps=0.5,                        # DBSCAN epsilon
    min_samples=5,                  # Min samples per cluster
    fingerprint_type='morgan',
    radius=2,
    nbits=2048,
    test_size=0.2,
    random_state=42
)

# Access results
train_idx = splits['train_indices']
test_idx = splits['test_indices']
info = splits['split_info']

print(f"Clusters: {info['n_clusters']}")
print(f"Silhouette score: {info['silhouette_score']:.3f}")
print(f"Feature source: {info['feature_source']}")
```

#### Implementation Details

```python
def feature_clustering_split(
    dataset: MolecularDataset,
    test_size: float = 0.2,
    clustering_algorithm: Literal['kmeans', 'hierarchical', 'dbscan'] = 'kmeans',
    n_clusters: Optional[int] = None,
    auto_select_k: bool = True,
    features: Optional[np.ndarray] = None,
    use_descriptors: bool = False,
    fingerprint_type: str = 'morgan',
    radius: int = 2,
    nbits: int = 2048,
    random_state: int = 42,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Feature-based clustering + leave-cluster-out splitting.

    Supports arbitrary molecular representations beyond fingerprints.
    """
```

**Key RDKit Descriptors Computed** (when `use_descriptors=True`):
- MW, LogP, TPSA, NumRotatableBonds
- NumHBondDonors, NumHBondAcceptors
- NumAromaticRings, FractionCSP3
- MolMR, BalabanJ, BertzCT, Chi0v
- HallKierAlpha, Kappa1, LabuteASA
- PEOE_VSA1, SMR_VSA1

#### Feature Clustering vs Butina

| Feature | Feature Clustering | Butina |
|---------|-------------------|--------|
| **Feature Input** | Arbitrary (3D, embeddings, descriptors) | Fingerprints only |
| **Similarity Metric** | Euclidean distance (after standardization) | Tanimoto similarity |
| **Clustering Algorithm** | K-means / Hierarchical / DBSCAN | Sphere exclusion (Butina) |
| **Cluster Count** | Manual or auto-optimized (Silhouette) | Auto (threshold-based) |
| **Use Case** | Non-fingerprint representations | Fingerprint-based validation |
| **Flexibility** | High (algorithm choice, features) | Moderate (fingerprint + threshold) |

**When to Use Feature Clustering**:
- ✅ Using 3D molecular embeddings (e.g., Boltz-2 structure predictions)
- ✅ Language model representations (ChemBERTa, MolFormer)
- ✅ Quantum chemical descriptors or docking features
- ✅ Need control over clustering algorithm

**When to Use Butina**:
- ✅ Standard fingerprint-based validation
- ✅ Need Tanimoto similarity specifically
- ✅ Simpler, domain-specific clustering

```python
# Typical workflow: Boltz-2 3D embeddings + Feature Clustering
from polyglotmol.representations import Boltz2Embedder

# Generate 3D structure embeddings
embedder = Boltz2Embedder()
embeddings = embedder.generate(dataset)  # Shape: (n, 768)

# Cluster-based split in embedding space
splits = feature_clustering_split(
    dataset=dataset,
    features=embeddings,
    clustering_algorithm='kmeans',
    auto_select_k=True,
    test_size=0.2
)

# Result: Test set contains molecules from distinct 3D structure clusters
```

#### Advantages

```{admonition} Why Use Feature Clustering Split?
:class: tip

**1. Representation Flexibility**: Works with any molecular representation (not limited to fingerprints)

**2. Algorithm Control**: Choose clustering algorithm based on data distribution

**3. Auto K-Selection**: Silhouette score optimization removes manual tuning

**4. Quality Assurance**: Three independent metrics (Silhouette, CH, DB) validate clustering

**5. Standardization**: Auto feature scaling ensures fair distance computation

**6. Advanced Representations**: Ideal for 3D, language models, quantum features
```

### 8. User-Provided Splits

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
