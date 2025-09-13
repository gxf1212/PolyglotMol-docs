# Protein Sequence Tokenizer API

API reference for protein sequence tokenization functionality.

## Overview

This module provides tokenization capabilities for protein sequences, converting amino acid sequences into tokens suitable for machine learning models.

## Classes

### ProteinSequenceTokenizer

```{eval-rst}
.. autoclass:: polyglotmol.representations.protein.sequential.ProteinSequenceTokenizer
   :members:
   :show-inheritance:
```

### ESMTokenizer

```{eval-rst}
.. autoclass:: polyglotmol.representations.protein.sequential.ESMTokenizer
   :members:
   :show-inheritance:
```

## Functions

### Tokenization Utilities

```{eval-rst}
.. automodule:: polyglotmol.representations.protein.sequential.tokenizer
   :members:
   :show-inheritance:
```

## Usage Examples

### Basic Tokenization

```python
from polyglotmol.representations.protein.sequential import ProteinSequenceTokenizer

# Create tokenizer
tokenizer = ProteinSequenceTokenizer()

# Tokenize protein sequence
sequence = "MKFLILLFNILCLFPVLAADNHKDKAMEALQLSSLRSHPKSAEEHKKQQELQHQRQQERLSQHRQQMQRQSQQLLQ"
tokens = tokenizer.tokenize(sequence)

# Convert to token IDs
token_ids = tokenizer.encode(sequence)
```

### ESM Model Integration

```python
from polyglotmol.representations.protein.sequential import ESMTokenizer

# Create ESM tokenizer
esm_tokenizer = ESMTokenizer(model_name="esm2_t6_8M_UR50D")

# Tokenize for ESM model
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
tokens = esm_tokenizer.tokenize(sequence)
```

## See Also

- {doc}`../../../../usage/representations/protein/sequence/tokenizer` - Usage guide
- {doc}`../structure/index` - Protein structure representations