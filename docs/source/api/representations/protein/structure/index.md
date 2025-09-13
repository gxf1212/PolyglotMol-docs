# Protein Structure API

API reference for protein structure-based representations.

## Overview

This module provides structure-based protein representations including MaSIF surface fingerprints, geometric descriptors, and structural embeddings.

## Classes

### MaSIFFeaturizer

```{eval-rst}
.. autoclass:: polyglotmol.representations.protein.structure.MaSIFFeaturizer
   :members:
   :show-inheritance:
```

### ProteinSurfaceFeaturizer

```{eval-rst}
.. autoclass:: polyglotmol.representations.protein.structure.ProteinSurfaceFeaturizer
   :members:
   :show-inheritance:
```

### ProteinGeometryFeaturizer

```{eval-rst}
.. autoclass:: polyglotmol.representations.protein.structure.ProteinGeometryFeaturizer
   :members:
   :show-inheritance:
```

## Utility Functions

```{eval-rst}
.. automodule:: polyglotmol.representations.protein.structure.utils
   :members:
   :show-inheritance:
```

## Usage Examples

### Basic MaSIF Usage

```python
from polyglotmol.representations.protein.structure import MaSIFFeaturizer

# Create MaSIF featurizer
masif = MaSIFFeaturizer()

# Process protein structure
features = masif.featurize("path/to/protein.pdb")
```

### Surface Analysis

```python
from polyglotmol.representations.protein.structure import ProteinSurfaceFeaturizer

# Create surface featurizer
surface = ProteinSurfaceFeaturizer(patch_size=1000)

# Generate surface features
surface_features = surface.featurize("protein.pdb")
```

## See Also

- {doc}`../../../../usage/representations/protein/structure/masif` - MaSIF usage guide
- {doc}`../sequential/tokenizer` - Protein sequence representations