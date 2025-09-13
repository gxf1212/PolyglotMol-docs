# Protein-Ligand Representations

Specialized 3D fingerprints for protein-ligand complexes, designed for structure-based drug design and binding affinity prediction. These representations capture topological features, pharmacophore interactions, and molecular fields that traditional 2D fingerprints cannot detect.

PolyglotMol provides unified access to five state-of-the-art 3D fingerprints with consistent API, flexible input handling, and optimized batch processing for high-throughput virtual screening.

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🧬 **Topology Net 3D**
:link: topology_net_3d
:link-type: doc

Persistent homology capturing molecular topology
:::

:::{grid-item-card} 🔗 **Enhanced SPLIF**
:link: splif_enhanced
:link-type: doc

Pharmacophore interaction patterns
:::

:::{grid-item-card} ⚡ **3D QSAR Fields**
:link: qsar_3d_fields
:link-type: doc

CoMFA/CoMSIA-style molecular fields
:::

:::{grid-item-card} 📏 **Distance Matrix**
:link: distance_matrix
:link-type: doc

Binned distance interactions (protein reuse)
:::

:::{grid-item-card} 🎯 **Atomic Distance Matrix**
:link: ligand_residue_matrix
:link-type: doc

Full atomic resolution distance matrix
:::

::::

## Quick Start

```python
import polyglotmol as pm
import numpy as np

# Three ways to provide molecular input
# Method 1: Direct SMILES string
featurizer = pm.get_featurizer('topology_net_3d')
fp_smiles = featurizer.featurize('CCO')
print(fp_smiles.shape)  # (512,)
print(fp_smiles.dtype)  # float64

# Method 2: PolyglotMol Molecule object
mol = pm.Molecule.from_smiles('CCO')
fp_mol = featurizer.featurize(mol)
print(np.array_equal(fp_smiles, fp_mol))  # True

# Method 3: RDKit Mol object
rdkit_mol = pm.mol_from_input('CCO')
fp_rdkit = featurizer.featurize(rdkit_mol)
print(np.array_equal(fp_mol, fp_rdkit))  # True
```

## Available Featurizers

:::{list-table} **Protein-Ligand 3D Fingerprints**
:header-rows: 1
:widths: 30 15 15 40

* - Featurizer Key
  - Type
  - Size
  - Description
* - `topology_net_3d`
  - Float
  - 512
  - Persistent homology topological features
* - `splif_enhanced`
  - Count
  - 2048
  - Pharmacophore interaction patterns
* - `qsar_3d_fields`
  - Float
  - 1000
  - Electrostatic, steric, hydrophobic fields
* - `protein_ligand_distance_matrix`
  - Float
  - 200
  - Binned distance interactions (optimized for single protein)
* - `ligand_residue_distance_matrix`
  - Float
  - 6000
  - Full atomic resolution distance matrix (30×200)
:::

```{tip}
Copy any featurizer key from the table above and use it directly with `pm.get_featurizer()`!
```

## Batch Processing

Process multiple molecules efficiently with automatic parallelization:

```python
# Sample dataset for structure-based screening
smiles_list = [
    'CC(=O)Oc1ccccc1C(=O)O',    # aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # caffeine  
    'CC(C)(C)NCC(c1ccc(O)cc1O)O',    # salbutamol
    'CCN(CC)C(=O)c1ccc2c(c1)ncn2C',  # theophylline
    'c1ccc2c(c1)c(c(=O)[nH]2)CC(=O)O'  # indole-3-acetic acid
]

featurizer = pm.get_featurizer('topology_net_3d')

# Sequential processing
fps_sequential = featurizer.featurize(smiles_list)
print(f"Output count: {len(fps_sequential)}")  # 5 molecules
print(f"Each feature shape: {fps_sequential[0].shape}")  # (512,)

# Parallel processing with featurize_many
fps_parallel = featurizer.featurize_many(smiles_list, n_jobs=4)

# With error handling
fps, errors = featurizer.featurize_many(
    smiles_list + ['invalid_smiles'],  # Add invalid molecule
    return_errors=True
)
print(f"Processed: {len(fps)}, Failed: {len(errors)}")
```

## Integration Examples

### Structure-Based Virtual Screening

```python
from polyglotmol.data import MolecularDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset with binding affinities
dataset = MolecularDataset.from_csv(
    "compounds.csv", 
    smiles_column="SMILES",
    target_column="binding_affinity"
)

# Add 3D features for structure-based analysis
dataset.add_features("topology_net_3d", n_workers=4)
dataset.add_features("splif_enhanced", n_workers=4) 
dataset.add_features("qsar_3d_fields", n_workers=4)

# Combine features for ensemble modeling (convert to arrays first)
combined_features = np.hstack([
    np.array(dataset.features["topology_net_3d"]),
    np.array(dataset.features["splif_enhanced"]), 
    np.array(dataset.features["qsar_3d_fields"])
])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(combined_features, dataset.targets["binding_affinity"])

# Evaluate performance
predictions = model.predict(combined_features)
r2 = r2_score(dataset.targets["binding_affinity"], predictions)
print(f"Combined 3D fingerprints R²: {r2:.3f}")
```

---

For detailed class and function references, see the full API documentation:
{doc}`/api/representations/index`.

```{toctree}
:maxdepth: 1
:hidden:

topology_net_3d
splif_enhanced
qsar_3d_fields
distance_matrix
ligand_residue_matrix
```

