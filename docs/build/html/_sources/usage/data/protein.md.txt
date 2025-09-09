# Protein

The `Protein` class in PolyglotMol is a comprehensive container for protein data, designed to handle sequences, structures, and various protein-specific operations seamlessly. It provides a unified interface for working with raw sequences, structure files, and data from online databases like RCSB PDB, AlphaFold DB, and UniProt.

## Overview

The `Protein` class simplifies protein data handling with:
- **Multiple input formats**: sequences, FASTA, PDB/mmCIF files, database IDs
- **Automatic structure prediction**: ESMFold integration
- **Structure repair**: Fix missing atoms/residues with PDBFixer
- **Smart caching**: Avoid redundant downloads and computations
- **Comprehensive residue handling**: Support for non-standard and simulation-specific amino acids
- **BioPython integration**: Familiar API for structure manipulation

## Installation

```bash
# Optional dependencies for advanced features
pip install biopython       # For structure parsing
conda install -c conda-forge pdbfixer  # For structure repair
```

## Quick Start

```python
from polyglotmol.data import Protein

# From sequence
protein = Protein.from_sequence("MKTAYIAKQRQISFVKSHFSRQ")

# From PDB database
protein = Protein.from_pdb_id("1CRN")

# From AlphaFold database
protein = Protein.from_alphafold_id("P00533")

# From file
protein = Protein.from_file("/path/to/protein.pdb")

# Auto-detection
protein = Protein.from_input("MKTAYIAKQRQISFVKSHFSRQ")  # Detects as sequence
protein = Protein.from_input("1CRN")  # Detects as PDB ID
```

## Creating Protein Objects

:::::{tab-set}

::::{tab-item} From Sequences
```python
from polyglotmol.data import Protein

# Basic sequence input
protein = Protein.from_sequence("MKTAYIAKQRQISFVKSHFSRQ")
print(protein.get_sequence())
# Output: MKTAYIAKQRQISFVKSHFSRQ

# With automatic structure prediction
protein = Protein.from_sequence(
    "MKTAYIAKQRQISFVKSHFSRQ",
    predict_structure=True  # Uses ESMFold
)
print(protein.has_structure())
# Output: True

# From FASTA string
fasta_str = """>sp|P01308|INS_HUMAN Insulin
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"""
protein = Protein.from_fasta_string(fasta_str)
print(protein.get_sequence()[:20])
# Output: MALWMRLLPLLALLALWGPD
```
::::

::::{tab-item} From Structure Files
```python
# From PDB file with automatic repair
protein = Protein.from_pdb_file(
    "/path/to/protein.pdb",
    fix_structure=True  # Fix missing atoms
)

# From mmCIF file
protein = Protein.from_cif_file("/path/to/structure.cif")

# Access structure information
structure = protein.get_structure()
print(f"Structure ID: {structure.id}")
print(f"Number of models: {len(structure)}")
# Output: Structure ID: my_protein
# Output: Number of models: 1
```
::::

::::{tab-item} From Databases
```python
# From RCSB PDB (automatically cached)
protein = Protein.from_pdb_id("1CRN", fix_structure=True)
print(f"Sequence length: {len(protein.get_sequence())}")
# Output: Sequence length: 46

# From AlphaFold (predicted structure)
protein = Protein.from_alphafold_id("P00533")  # Human EGFR
print(protein.metadata['source'])
# Output: AlphaFold DB

# From UniProt (sequence only)
protein = Protein.from_uniprot_id("P01308")  # Human insulin
print(protein.metadata['protein_name'])
# Output: Insulin
```
::::

::::{tab-item} Auto-detection
```python
# The from_input method intelligently detects the input type
examples = [
    "MKTAYIAKQRQISFVKSHFSRQ",  # Sequence
    "1CRN",                     # PDB ID
    "P00533",                   # UniProt ID
    "/path/to/protein.pdb",     # File path
    ">test\nMKTAYIAKQRQ"       # FASTA string
]

for data in examples:
    protein = Protein.from_input(data)
    print(f"Input: {data[:20]}... → Type detected")
```
::::

:::::

## Working with Sequences

```python
# Get sequence from any source
protein = Protein.from_pdb_id("1UBQ")
sequence = protein.get_sequence()
print(f"Sequence: {sequence[:30]}...")
# Output: Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKI...

# Get sequences for all chains
chain_sequences = protein.get_chain_sequences()
for chain_id, seq in chain_sequences.items():
    print(f"Chain {chain_id}: {seq[:20]}... (length: {len(seq)})")
# Output: Chain A: MQIFVKTLTGKTITLEVEPS... (length: 76)

# Handle non-standard residues automatically
# MSE (selenomethionine) → M, PTR (phosphotyrosine) → Y, etc.
```

## Structure Operations

### Structure Prediction

```python
# Predict structure from sequence
protein = Protein.from_sequence("MKTAYIAKQRQISFVKSHFSRQ")

# Predict structure (cached automatically)
success = protein.predict_structure(method='esmfold')
print(f"Structure predicted: {success}")
# Output: Structure predicted: True

# Access the predicted structure
structure = protein.get_structure()
print(f"Number of atoms: {len(list(structure.get_atoms()))}")
# Output: Number of atoms: 324

# Force re-prediction
protein.predict_structure(method='esmfold', force=True)
```

### Structure Repair

```python
# Fix missing atoms/residues
protein = Protein.from_pdb_id("1ABC")  # Structure with gaps

# Check if repair needed
if protein.metadata.get('has_missing_atoms'):
    success = protein.fix_structure(method='pdbfixer')
    print(f"Structure fixed: {success}")
    
    # Check repair metadata
    fix_info = protein.metadata.get('structure_fixed', {})
    print(f"Missing residues fixed: {fix_info.get('missing_residues', 0)}")
    print(f"Missing atoms fixed: {fix_info.get('missing_atoms', 0)}")
```

## File I/O and Caching

### Saving Files

```python
# Save in different formats
protein.to_file("output.pdb", format="pdb")
protein.to_file("output.cif", format="cif")
protein.to_file("output.fasta", format="fasta")

# Convert to FASTA string
fasta_str = protein.to_fasta()
print(fasta_str)
# Output: >protein_id description
# Output: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
```

### Automatic Caching

```python
# First call downloads from database
protein1 = Protein.from_pdb_id("1CRN")  # Downloads from RCSB

# Second call loads from cache (fast!)
protein2 = Protein.from_pdb_id("1CRN")  # Loads from .polyglotmol_cache/

# Custom cache directory
protein = Protein.from_pdb_id(
    "1UBQ",
    cache_dir="/my/custom/cache"
)

# Structure predictions are also cached by sequence hash
seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV"
protein1 = Protein.from_sequence(seq)
protein1.predict_structure()  # Slow (prediction)

protein2 = Protein.from_sequence(seq)
protein2.predict_structure()  # Fast (cached)
```

## Handling Non-standard Residues

:::{admonition} Comprehensive Residue Support
:class: tip

The Protein class automatically handles various non-standard residue names commonly found in PDB files from simulations or experiments.
:::

```python
# Simulation-specific residues are mapped automatically
# HID, HIE, HIP → H (histidine states)
# CYX → C (disulfide cysteine)
# MSE → M (selenomethionine)

protein = Protein.from_pdb_file("simulation.pdb")
sequence = protein.get_sequence()
# Non-standard residues automatically converted to standard codes

# Access the residue mapping tables
from polyglotmol.data import (
    STANDARD_AMINO_ACIDS,
    SIMULATION_RESIDUE_MAPPING,
    NON_STANDARD_AMINO_ACIDS
)

print(f"Standard amino acids: {len(STANDARD_AMINO_ACIDS)}")
print(f"Simulation mappings: {len(SIMULATION_RESIDUE_MAPPING)}")
print(f"Non-standard mappings: {len(NON_STANDARD_AMINO_ACIDS)}")
```

## BioPython Integration

```python
from Bio import SeqIO
from Bio.PDB import PDBParser

# Create from BioPython SeqRecord
record = SeqIO.read("protein.fasta", "fasta")
protein = Protein(seqrecord=record)

# Create from BioPython Structure
parser = PDBParser()
bio_structure = parser.get_structure("test", "protein.pdb")
protein = Protein(structure=bio_structure)

# Access BioPython objects
structure = protein.get_structure()
if structure:
    # Use familiar BioPython API
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue
                    print(f"{residue.resname} {residue.id[1]}")
```

## Batch Processing

```python
from polyglotmol.data import proteins_from_fasta, fetch_multiple_from_pdb

# Load multiple proteins from FASTA
proteins = proteins_from_fasta("sequences.fasta")
for protein in proteins:
    seq = protein.get_sequence()
    print(f"Protein {protein.metadata.get('fasta_header')}: {len(seq)} residues")

# Fetch multiple structures from PDB
pdb_ids = ["1CRN", "1UBQ", "2GB1"]
proteins = fetch_multiple_from_pdb(
    pdb_ids,
    fix_structure=True,
    cache_dir="./pdb_cache"
)

for pdb_id, protein in proteins.items():
    if protein:
        print(f"{pdb_id}: {len(protein.get_sequence())} residues")
```

## Advanced Features

### Metadata Access

```python
# Get all metadata including PDB headers
metadata = protein.get_metadata()
print(f"Available metadata: {list(metadata.keys())}")

# Access specific metadata
if 'pdb_header' in metadata:
    header = metadata['pdb_header']
    print(f"Resolution: {header.get('resolution')} Å")
    print(f"R-free: {header.get('rfree')}")
    
# Structure prediction metadata
if 'structure_prediction' in metadata:
    pred_info = metadata['structure_prediction']
    print(f"Method: {pred_info['method']}")
    print(f"Date: {pred_info['date']}")
```

### Error Handling

```python
# Invalid sequences are cleaned automatically
protein = Protein.from_sequence("MKTAY@#IAKRQ")  # @ and # replaced with X
print(protein.get_sequence())
# Output: MKTAYXXIAKRQ

# Network errors handled gracefully
try:
    protein = Protein.from_pdb_id("XXXX")  # Invalid ID
except Exception as e:
    print(f"Error: {e}")
    # Failed downloads don't create corrupt cache files
```

## Feature Summary

| Feature | Description |
|---------|-------------|
| **Input Formats** | Sequences, FASTA, PDB, mmCIF, database IDs |
| **Databases** | RCSB PDB, AlphaFold DB, UniProt |
| **Structure Prediction** | ESMFold (automatic caching) |
| **Structure Repair** | PDBFixer integration |
| **Residue Handling** | Standard, non-standard, simulation-specific |
| **Caching** | Automatic for downloads and predictions |
| **BioPython** | Full integration with BioPython objects |
| **Batch Processing** | Multi-sequence FASTA, multiple PDB fetching |

## API Reference

- {doc}`/api/data/protein` - Full API documentation for the Protein class
- {doc}`/api/data/io` - InputType definitions and utilities

## External Resources

- [BioPython Documentation](https://biopython.org)
- [RCSB PDB](https://www.rcsb.org) - Protein structure database
- [AlphaFold Database](https://alphafold.ebi.ac.uk) - Predicted structures
- [UniProt](https://www.uniprot.org) - Protein sequence database
- [ESMFold](https://esmatlas.com/about) - Structure prediction
- [PDBFixer](https://github.com/openmm/pdbfixer) - Structure repair tool

```{toctree}
:maxdepth: 1
:hidden:
```