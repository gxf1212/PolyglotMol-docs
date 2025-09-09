# Installation

PolyglotMol is designed to be a modular toolkit. You can install a lightweight core and then add optional dependencies to enable specific functionalities like protein analysis, advanced machine learning models, or molecular dynamics.

The package requires **Python \>= 3.9**.

-----

## Installation Methods

We recommend using `conda` to manage complex cheminformatics dependencies like RDKit, but installation with `pip` is also fully supported.

### Method 1: Conda (Recommended)

Using `conda` is the most reliable way to install the core cheminformatics backends, as `conda-forge` provides pre-compiled binaries for all major operating systems.

```bash
# 1. Create and activate a new conda environment (recommended)
conda create -n polyglotmol_env python=3.9 -y
conda activate polyglotmol_env

# 2. Install the core RDKit backend from conda-forge
conda install -c conda-forge rdkit -y

# 3. Install PolyglotMol using pip
# This will install the core package and its essential dependencies (numpy, pandas).
pip install polyglotmol

# 4. Install optional functionalities as needed
# For example, to add support for protein featurizers and ML models:
pip install polyglotmol[protein,models]
```

### Method 2: pip

You can install PolyglotMol directly from PyPI.

#### Basic Installation

This installs the core package with its minimal dependencies (`numpy`, `pandas`, `rdkit-pypi`).

```bash
pip install polyglotmol
```

#### Installation with Extras

To use advanced features, you need to install "extras." This is done by adding brackets `[]` with the name of the feature group to the pip install command. You can install multiple groups at once.

```bash
# Install with a single optional group (e.g., for ML models)
pip install polyglotmol[models]

# Install multiple groups at once
pip install polyglotmol[deepchem,md]

# Install a convenience group that includes several related packages
pip install polyglotmol[cheminformatics_base]

# Install all optional dependencies
pip install polyglotmol[all]
```

-----

## Core and Optional Dependencies

### Core Dependencies

These are installed automatically with `pip install polyglotmol`:

  - [NumPy](https://numpy.org/)
  - [Pandas](https://pandas.pydata.org/)
  - [RDKit](https://www.rdkit.org/)

### Optional Dependency Groups (`extras`)

The following table lists the main optional groups and the key functionalities they enable.

| Installation Command                     | Key Libraries Installed                               | Functionality Enabled                                      |
| :--------------------------------------- | :---------------------------------------------------- | :--------------------------------------------------------- |
| `pip install polyglotmol[models]`        | `scikit-learn`, `xgboost`, `lightgbm`, `catboost`       | Standard and gradient-boosted models for QSAR/ML tasks.    |
| `pip install polyglotmol[protein]`       | `biopython`, `fair-esm`, `transformers`, `tokenizers` | Protein featurizers (PLMs) and structural biology tools.   |
| `pip install polyglotmol[cheminformatics_ml]` | `chemprop`, `mol2vec`                            | Advanced ML models like Chemprop and Mol2Vec embeddings.   |
| `pip install polyglotmol[deepchem]`      | `deepchem`                                            | Featurizers and models from the DeepChem library.          |
| `pip install polyglotmol[mordred]`       | `mordred`                                             | Calculation of a large set of 2D & 3D molecular descriptors. |
| `pip install polyglotmol[spatial]`       | `unimol-tools`, `dscribe`, `ase`                      | 3D spatial representations like Coulomb Matrices and Uni-Mol. |
| `pip install polyglotmol[md]`            | `mdanalysis`                                          | Analysis of molecular dynamics trajectories.               |
| `pip install polyglotmol[cdk]`           | `CDK-pywrapper`                                       | CDK-based fingerprints (requires **Java \>= 11**).          |
| `pip install polyglotmol[molfeat]`       | `molfeat`                                             | Various fingerprints and representations via Molfeat.      |
| `pip install polyglotmol[drawing]`       | `matplotlib`, `seaborn`                               | Molecule and data visualization utilities.                 |
| `pip install polyglotmol[openbabel]`     | `openbabel-wheel`                                     | Support for Open Babel via Pybel.                          |

-----

## Developer Installation

If you want to contribute to PolyglotMol, you should clone the repository and install it in editable mode with the development dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/gxf1212/polyglotmol.git
cd polyglotmol

# 2. Create and activate a conda environment (recommended)
conda create -n polyglotmol_dev python=3.9 -y
conda activate polyglotmol_dev
conda install -c conda-forge rdkit -y # Install base dependencies

# 3. Install the package in editable mode with all development extras
pip install -e .[dev,all]

# 4. Set up pre-commit hooks (optional, but recommended)
pre-commit install
```

This will install the package in a way that your changes to the source code are immediately reflected, and it will also install all testing and linting tools like `pytest` and `ruff`.