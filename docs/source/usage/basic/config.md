# Configuration

```{toctree}
:maxdepth: 1
:hidden:
```

PolyglotMol automatically manages model caching for pre-trained models used by advanced featurizers (PLMs, UniMol, etc.). Models are downloaded once and reused from local cache.

## Cache Priority

PolyglotMol determines cache locations in this order:

1. **Environment variables** (highest priority)
2. **API settings** via {func}`~polyglotmol.config.set_cache_dir`  
3. **Default paths** under `~/.cache/polyglotmol/`

## Environment Variables

Set these before importing PolyglotMol:

```bash
# PyTorch Hub models (ESM, CARP)
export TORCH_HOME=/my/custom/torch/cache

# Hugging Face models (ProtT5, Ankh, PepBERT)  
export HF_HOME=/my/custom/hf/cache

# Optional: Use mirror for faster downloads
export HF_ENDPOINT=https://hf-mirror.com
```

## Default Paths

If environment variables aren't set, PolyglotMol uses:
- `~/.cache/polyglotmol/torch_hub/` for PyTorch Hub
- `~/.cache/polyglotmol/huggingface_hub/` for Hugging Face

## Programmatic Control

```python
from polyglotmol.config import set_cache_dir, get_cache_dir

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

PolyglotMol logs effective paths on import:

```python
import polyglotmol
# INFO: [PolyglotMol Settings] Effective TORCH_HOME: /home/user/.cache/polyglotmol/torch_hub
# INFO: [PolyglotMol Settings] Effective HF_HOME: /home/user/.cache/polyglotmol/huggingface_hub
# INFO: [PolyglotMol Settings] Using HF_ENDPOINT (mirror): https://hf-mirror.com (if set)
```

## Model Loading Example

```python
from polyglotmol.representations.protein.sequence.plm import ProteinLanguageModelFeaturizer

# Models are automatically downloaded to configured cache
featurizer = ProteinLanguageModelFeaturizer(
    model_name="Rostlab/prot_t5_xl_half_uniref50",
    model_type="t5",
    batch_size=8
)

# First run downloads model (~892MB for ProtT5-XL)
# Subsequent runs load from cache instantly
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

Once models are cached, PolyglotMol works offline:

```python
# After initial download, this works without internet
from polyglotmol.representations.spatial.unimol import UniMolFeaturizer

unimol = UniMolFeaturizer()
# Loads from local cache at TORCH_HOME or HF_HOME
```

## API Reference

- {func}`~polyglotmol.config.set_cache_dir` - Set cache directory
- {func}`~polyglotmol.config.get_cache_dir` - Get current cache path
- {attr}`~polyglotmol.config.EFFECTIVE_TORCH_HOME` - Active PyTorch cache
- {attr}`~polyglotmol.config.EFFECTIVE_HF_HOME` - Active HuggingFace cache

## Related Links

- [PyTorch Hub Documentation](https://pytorch.org/docs/stable/hub.html)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/huggingface_hub/)