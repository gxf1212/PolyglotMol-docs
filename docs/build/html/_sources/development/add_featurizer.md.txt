# Adding a New Featurizer to PolyglotMol

## Overview

This guide provides a complete workflow for adding new molecular featurizers to PolyglotMol. Following these standards ensures consistency, proper integration, and maintainability.

```mermaid
graph LR
    A[Choose Location] --> B[Implement Class]
    B --> C[Register Featurizer]
    C --> D[Update Imports]
    D --> E[Add Tests]
    E --> F[Document]
```

## Prerequisites

- Understanding of the molecular representation you're implementing
- Required dependencies installed (RDKit, DeepChem, etc.)
- Familiarity with NumPy arrays and Python type hints

---

## Step 1: Choose the Modality and Location

### 1.1 Identify Representation Type

Determine where your featurizer belongs:

| Type | Description | Location | Examples |
|------|-------------|----------|----------|
| **Fingerprints** | Binary/count vectors | `/fingerprints/` | Morgan, MACCS, ECFP |
| **Descriptors** | Molecular properties | `/descriptors/` | RDKit descriptors, Mordred |
| **Spatial** | 3D structure-based | `/spatial/` | Coulomb matrix, UniMol |
| **Graph** | Graph representations | `/graph/` | GraphConv, Weave |
| **Sequential** | String/sequence-based | `/sequential/` | SMILES-BERT, ChemBERTa |
| **Temporal** | Time-series data | `/temporal/` | MD trajectories |
| **Protein** | Protein-specific | `/protein/` | ESM, ProtT5 |

### 1.2 Create File Structure

```bash
# Navigate to appropriate directory
cd src/polyglotmol/representations/fingerprints/

# Create your new featurizer file
touch my_new_fp.py
```

---

## Step 2: Implement the Featurizer Class

### 2.1 Basic Class Structure

```python
# src/polyglotmol/representations/fingerprints/my_new_fp.py
# -*- coding: utf-8 -*-
"""
My New Fingerprint Implementation.

Detailed description of what this fingerprint represents and its use cases.
"""
import logging
from typing import Any, Optional, Tuple, Type
import numpy as np

# External dependencies
from rdkit import Chem  # If needed

# PolyglotMol imports
from ..utils.base import BaseFeaturizer
from ..utils.exceptions import FeaturizationError, DependencyNotFoundError
from ..utils.registry import register_featurizer
from ...data.io import InputType

logger = logging.getLogger(__name__)


class MyNewFeaturizer(BaseFeaturizer):
    """
    My new molecular featurizer.
    
    Parameters
    ----------
    n_bits : int, default=2048
        Number of bits in the fingerprint
    radius : int, default=2
        Radius for circular fingerprints
    
    Attributes
    ----------
    OUTPUT_SHAPE : tuple
        Expected shape of the output
    """
    
    # Define expected input and output
    EXPECTED_INPUT_TYPE = (InputType.SMILES, InputType.RDKIT_MOL)
    OUTPUT_SHAPE = (2048,)  # Will be updated in __init__ if dynamic
    
    def __init__(self, n_bits: int = 2048, radius: int = 2, **kwargs):
        """Initialize the featurizer."""
        # ALWAYS call super().__init__ with all parameters
        super().__init__(n_bits=n_bits, radius=radius, **kwargs)
        
        # Update output shape if it depends on parameters
        self.OUTPUT_SHAPE = (n_bits,)
        
        # Set description if not provided
        if not self.description:
            self.description = f"My fingerprint (radius={radius}, {n_bits} bits)"
        
        # Check dependencies
        self._check_dependencies()
        
        # DON'T store non-pickleable objects here!
        # self._generator = SomeGenerator()  # ❌ BAD
        
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import required_library
        except ImportError:
            raise DependencyNotFoundError(
                "Required library not found. Install with: pip install required-library"
            )
    
    def _featurize(self, data_point: Any, **kwargs) -> np.ndarray:
        """
        Generate fingerprint for a single molecule.
        
        Parameters
        ----------
        data_point : RDKit Mol or SMILES string
            Single molecule to featurize
        **kwargs : dict
            Additional parameters (merged from __init__ and featurize call)
            
        Returns
        -------
        np.ndarray
            Fingerprint as numpy array
            
        Raises
        ------
        FeaturizationError
            If featurization fails for this input
        """
        # Extract parameters
        n_bits = kwargs.get('n_bits', self.init_kwargs.get('n_bits', 2048))
        radius = kwargs.get('radius', self.init_kwargs.get('radius', 2))
        
        # Handle different input types
        if isinstance(data_point, str):
            mol = Chem.MolFromSmiles(data_point)
            if mol is None:
                raise FeaturizationError(f"Invalid SMILES: {data_point}")
        else:
            mol = data_point
            
        try:
            # Your featurization logic here
            # Example: generate fingerprint
            from rdkit.Chem import AllChem
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            
            # Convert to numpy array
            arr = np.zeros((n_bits,), dtype=np.uint8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
            
        except Exception as e:
            logger.error(f"Featurization failed: {e}")
            raise FeaturizationError(f"Failed to generate fingerprint: {str(e)}")
```

### 2.2 Alternative: Using Base Classes for Common Patterns

For RDKit fingerprints, inherit from `BaseRDKitFingerprint`:

```python
from .rdkit import BaseRDKitFingerprint

class MyRDKitFingerprint(BaseRDKitFingerprint):
    def _create_generator_func(self) -> Callable:
        """Return the RDKit fingerprint generator function."""
        from functools import partial
        from rdkit.Chem import AllChem
        
        return partial(
            AllChem.GetMorganFingerprintAsBitVect,
            radius=self.init_kwargs.get('radius', 2),
            nBits=self.init_kwargs.get('n_bits', 2048)
        )
```

---

## Step 3: Register the Featurizer

### 3.1 Registration with Shape Information

Always include shape information when registering:

```python
# Method 1: Using decorator (single configuration)
@register_featurizer(
    "my_new_fp",
    description="My new fingerprint with default settings",
    category="fingerprints/mynewtype",
    shape=(2048,)  # ← Important!
)
class MyNewFeaturizer(BaseFeaturizer):
    # ... class implementation ...
```

```python
# Method 2: Multiple configurations (recommended)
# Place at the end of the file
register_featurizer(
    "my_fp_r2_2048",
    cls=MyNewFeaturizer,
    default_kwargs={"radius": 2, "n_bits": 2048},
    description="My fingerprint (radius 2, 2048 bits)",
    category="fingerprints/mynewtype",
    shape=(2048,)
)

register_featurizer(
    "my_fp_r3_1024",
    cls=MyNewFeaturizer,
    default_kwargs={"radius": 3, "n_bits": 1024},
    description="My fingerprint (radius 3, 1024 bits)",
    category="fingerprints/mynewtype",
    shape=(1024,)
)

register_featurizer(
    "my_fp_counts",
    cls=MyNewFeaturizer,
    default_kwargs={"radius": 2, "use_counts": True},
    description="My fingerprint with counts",
    category="fingerprints/mynewtype",
    shape="dynamic"  # For variable-length outputs
)
```

### 3.2 Shape Guidelines

| Output Type | Shape Value | Example |
|-------------|-------------|---------|
| Fixed size | `(size,)` | `(2048,)` |
| Dynamic/variable | `"dynamic"` | Count fingerprints |
| 2D matrix | `(rows, cols)` | `(100, 100)` |
| Multi-dimensional | `(d1, d2, d3, ...)` | `(224, 224, 3)` |

---

## Step 4: Update Sub-package Imports

### 4.1 Update `__init__.py`

```python
# src/polyglotmol/representations/fingerprints/__init__.py
"""
Fingerprint Representations Subpackage.

Dynamically imports all fingerprint modules to trigger registration.
"""
import logging
import importlib
import pkgutil

logger = logging.getLogger(__name__)

# Import specific modules
from . import rdkit
from . import cdk
from . import deepchem
from . import datamol
from . import my_new_fp  # ← Add your module here

# Or use automatic discovery (alternative approach)
# ... existing auto-discovery code ...
```

---

## Step 5: Add Comprehensive Tests

### 5.1 Test Structure

```python
# tests/representations/fingerprints/test_my_new_fp.py
import pytest
import numpy as np
import polyglotmol as pm
from polyglotmol.representations.utils.exceptions import (
    FeaturizationError, DependencyNotFoundError
)


class TestMyNewFingerprint:
    """Test suite for MyNewFeaturizer."""
    
    @pytest.fixture
    def valid_smiles(self):
        """Provide test molecules."""
        return ["CCO", "c1ccccc1", "CC(=O)O"]
    
    @pytest.fixture
    def invalid_smiles(self):
        """Provide invalid SMILES."""
        return ["invalid", "C(C", ""]
    
    def test_registration(self):
        """Test that featurizers are properly registered."""
        available = pm.list_available_featurizers()
        assert "my_fp_r2_2048" in available
        assert "my_fp_r3_1024" in available
        
    def test_instantiation(self):
        """Test featurizer instantiation."""
        # Default configuration
        feat1 = pm.get_featurizer("my_fp_r2_2048")
        assert feat1.init_kwargs["radius"] == 2
        assert feat1.init_kwargs["n_bits"] == 2048
        
        # Custom parameters
        feat2 = pm.get_featurizer("my_fp_r2_2048", n_bits=512)
        assert feat2.init_kwargs["n_bits"] == 512
        
    def test_shape_info(self):
        """Test shape information."""
        info = pm.get_featurizer_info("my_fp_r2_2048")
        assert info["shape"] == (2048,)
        assert info["shape_type"] == "fixed"
        
    def test_single_molecule(self, valid_smiles):
        """Test featurization of single molecules."""
        feat = pm.get_featurizer("my_fp_r2_2048")
        
        for smiles in valid_smiles:
            fp = feat.featurize(smiles)
            assert isinstance(fp, np.ndarray)
            assert fp.shape == (2048,)
            assert fp.dtype == np.uint8
            
    def test_invalid_input(self, invalid_smiles):
        """Test handling of invalid inputs."""
        feat = pm.get_featurizer("my_fp_r2_2048")
        
        for smiles in invalid_smiles:
            result = feat.featurize(smiles)
            assert result is None  # Should return None for invalid
            
    def test_batch_processing(self, valid_smiles, invalid_smiles):
        """Test batch featurization."""
        feat = pm.get_featurizer("my_fp_r2_2048")
        
        # Mix of valid and invalid
        all_smiles = valid_smiles + invalid_smiles
        results = feat.featurize_many(all_smiles)
        
        assert len(results) == len(all_smiles)
        assert all(isinstance(r, np.ndarray) for r in results[:3])
        assert all(r is None for r in results[3:])
        
    def test_parallel_processing(self, valid_smiles):
        """Test parallel featurization."""
        feat = pm.get_featurizer("my_fp_r2_2048")
        
        # Should work with multiple workers
        results = feat.featurize_many(valid_smiles * 10, n_workers=2)
        assert len(results) == len(valid_smiles) * 10
        assert all(isinstance(r, np.ndarray) for r in results)
        
    def test_error_handling(self):
        """Test error handling."""
        # Missing dependency
        with pytest.raises(DependencyNotFoundError):
            from polyglotmol.representations.fingerprints.my_new_fp import MyNewFeaturizer
            # Simulate missing dependency in __init__
            MyNewFeaturizer()
```

---

## Step 6: Documentation

### 6.1 Docstring Standards

Follow NumPy-style docstrings:

```python
def featurize(self, molecules, n_workers: int = 1, **kwargs):
    """
    Generate fingerprints for molecules.
    
    Parameters
    ----------
    molecules : str, RDKit Mol, or list
        Molecule(s) to featurize
    n_workers : int, default=1
        Number of parallel workers
    **kwargs : dict
        Additional parameters to override defaults
        
    Returns
    -------
    np.ndarray or list of np.ndarray
        Fingerprint(s) as numpy arrays
        
    Examples
    --------
    >>> feat = MyNewFeaturizer(n_bits=1024)
    >>> fp = feat.featurize("CCO")
    >>> fp.shape
    (1024,)
    """
```

### 6.2 Update Documentation Files

Add your featurizer to relevant documentation:
- `docs/source/api/representations/fingerprints/index.rst`
- `docs/source/usage/representations/fingerprints/examples.md`

---

## Quick Checklist

Before submitting your featurizer, ensure:

- [ ] **Class Implementation**
  - [ ] Inherits from appropriate base class
  - [ ] Sets `EXPECTED_INPUT_TYPE`
  - [ ] Sets `OUTPUT_SHAPE` (or updates in `__init__`)
  - [ ] Implements `_featurize` or `_create_generator_func`
  - [ ] Handles errors with `FeaturizationError`
  - [ ] No non-pickleable objects stored as instance attributes

- [ ] **Registration**
  - [ ] Unique name(s) chosen
  - [ ] `shape` parameter included
  - [ ] `category` parameter included
  - [ ] `description` provided
  - [ ] Default parameters set appropriately

- [ ] **Integration**
  - [ ] Added to sub-package `__init__.py`
  - [ ] Dependencies checked/documented

- [ ] **Testing**
  - [ ] Registration tests
  - [ ] Shape information tests
  - [ ] Single molecule tests
  - [ ] Batch processing tests
  - [ ] Parallel processing tests
  - [ ] Error handling tests

- [ ] **Documentation**
  - [ ] Class docstring complete
  - [ ] Method docstrings follow NumPy style
  - [ ] Usage examples provided

---

## Common Patterns and Examples

### Pattern 1: Featurizer with Model Loading

```python
class PretrainedModelFeaturizer(BaseFeaturizer):
    def __init__(self, model_name: str = "default", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._model = None  # Lazy loading
        
    def _ensure_model_loaded(self):
        """Load model on first use."""
        if self._model is None:
            from polyglotmol.config.models import load_model
            self._model = load_model(self.init_kwargs["model_name"])
            
    def _featurize(self, data_point: Any, **kwargs) -> np.ndarray:
        self._ensure_model_loaded()
        # Use self._model for featurization
```

### Pattern 2: Featurizer with Dynamic Shapes

```python
@register_featurizer(
    "my_count_fp",
    description="Count-based fingerprint",
    shape="dynamic"  # Variable length output
)
class CountFingerprint(BaseFeaturizer):
    OUTPUT_SHAPE = None  # Dynamic shape
    
    def get_output_info(self) -> Tuple[Type, Tuple[Optional[int], ...]]:
        """Return output type and shape information."""
        return (dict, ())  # Returns dict with variable size
```

### Pattern 3: Protein Featurizer

```python
from ..utils.base import BaseProteinFeaturizer
from ..utils.registry import register_protein_featurizer

@register_protein_featurizer(
    "my_protein_feat",
    description="My protein featurizer",
    shape=(768,),  # Per-sequence embedding
    per_residue_shape=(None, 768)  # Per-residue embeddings
)
class MyProteinFeaturizer(BaseProteinFeaturizer):
    EXPECTED_INPUT_TYPE = InputType.PROTEIN_SEQUENCE
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure your module is imported in the sub-package `__init__.py`
2. **Registration Not Found**: Check that registration happens at module level, not inside conditions
3. **Pickling Errors**: Don't store generators or models as instance attributes
4. **Shape Mismatches**: Verify OUTPUT_SHAPE matches actual output

### Getting Help

- Check existing implementations for examples
- Run tests with `pytest -xvs tests/representations/fingerprints/test_my_new_fp.py`
- Enable debug logging: `logging.getLogger('polyglotmol').setLevel(logging.DEBUG)`

---

