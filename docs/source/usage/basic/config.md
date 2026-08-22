# Configuration

```{toctree}
:maxdepth: 1
:hidden:
```

MolBlender automatically manages model caching for pre-trained models used by advanced featurizers (PLMs, UniMol, etc.). Models are loaded from local cache; downloads require `MOLBLENDER_ALLOW_MODEL_DOWNLOAD=1`.

## Cache Priority

MolBlender determines cache locations in this order:

1. **Environment variables** (highest priority)
2. **API settings** via {func}`~molblender.config.set_cache_dir`  
3. **Default paths** under `~/.cache/molblender/`

## Environment Variables

Set these before importing MolBlender:

```bash
# PyTorch Hub models (ESM, CARP)
export TORCH_HOME=/my/custom/torch/cache

# Hugging Face models (ProtT5, Ankh, PepBERT)  
export HF_HOME=/my/custom/hf/cache

# Optional: Use mirror for faster downloads
export HF_ENDPOINT=https://hf-mirror.com
```

## Default Paths

If environment variables aren't set, MolBlender uses:
- `~/.cache/molblender/torch_hub/` for PyTorch Hub
- `~/.cache/molblender/huggingface_hub/` for Hugging Face

## Programmatic Control

```python
from molblender.config import set_cache_dir, get_cache_dir

# Set custom cache directories
set_cache_dir("torch", "/data/models/torch")
set_cache_dir("hf", "/data/models/huggingface")

# Check current settings
print(f"PyTorch cache: {get_cache_dir('torch')}")
print(f"HuggingFace cache: {get_cache_dir('hf')}")
# Output: PyTorch cache: /data/models/torch
# Output: HuggingFace cache: /data/models/huggingface
```

## Verifying Settings

Use {func}`~molblender.config.get_cache_dir` to check current paths:

```python
from molblender.config import get_cache_dir

print(f"PyTorch cache: {get_cache_dir('torch')}")
print(f"HuggingFace cache: {get_cache_dir('hf')}")
```

## Model Loading Example

```python
from molblender.representations.protein.sequence.plm import ProteinLanguageModelFeaturizer

# Models are loaded from configured cache; downloads require
# MOLBLENDER_ALLOW_MODEL_DOWNLOAD=1
featurizer = ProteinLanguageModelFeaturizer(
    model_name="Rostlab/prot_t5_xl_half_uniref50",
    model_type="t5",
    batch_size=8
)

# Loads ProtT5-XL from local cache (~900MB model)
embeddings = featurizer.featurize(["MKTAYIAKQRQISFVKSHFSRQ"])
print(f"Embedding shape: {embeddings.shape}")
# Output: Embedding shape: (1, 1024)
```

## Disk Space Requirements

Common model sizes:
- ESM-2 (650M params): ~2.5GB
- ProtT5-XL: ~900MB  
- Ankh Large: ~1.5GB
- UniMol: ~300MB

## Offline Usage

Once models are cached, MolBlender works offline:

```python
# After initial download, this works without internet
from molblender.representations.spatial.unimol import UniMolCLSFeaturizer

unimol = UniMolCLSFeaturizer()
# Loads from local cache at TORCH_HOME or HF_HOME
```

## Mainland China / HuggingFace Mirror

If you are running MolBlender in mainland China, model downloads from the
default Hugging Face endpoint can be slow or unreliable. Set the standard
Hugging Face environment variables before using PLM or transformer-backed
representations:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/.cache/huggingface
```

Then run MolBlender normally:

```python
import molblender as mbl

featurizer = mbl.get_protein_featurizer("protT5")
embedding = featurizer("MKTAYIAKQRQISFVKSHFSRQ")
```

This is especially useful for:

- Protein language models (`esm2`, `protT5`, `ankh`)
- Transformer-based molecular representations
- Any representation that downloads model weights from Hugging Face

## API Reference

- {func}`~molblender.config.set_cache_dir` - Set cache directory programmatically
- {func}`~molblender.config.get_cache_dir` - Get current cache path (recommended for all new code)

> **Note:** New code should use `get_cache_dir("hf")` / `get_cache_dir("torch")` instead of importing `EFFECTIVE_HF_HOME` / `EFFECTIVE_TORCH_HOME` directly. The module-level constants are import-time snapshots; `get_cache_dir()` always returns the live value even after `set_cache_dir()` is called.

## Related Links

- [PyTorch Hub Documentation](https://pytorch.org/docs/stable/hub.html)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/huggingface_hub/)
