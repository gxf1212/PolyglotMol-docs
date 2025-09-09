# Machine Learning Models

PolyglotMol provides intelligent automated machine learning capabilities with multi-stage model screening, hyperparameter optimization, and comprehensive evaluation tools designed for molecular data.

## Introduction

The models module automates the entire ML pipeline for molecular property prediction, from feature selection to model deployment. Key capabilities include:

- **Automated Model Screening**: Test hundreds of model+representation combinations automatically
- **Multi-Stage Optimization**: Coarse → Fine → Adaptive hyperparameter tuning  
- **Parallel Execution**: Leverage multi-core systems and HPC clusters
- **Intelligent Selection**: Ensure representation diversity and avoid overfitting to single methods
- **Fault Tolerance**: Checkpoint progress and handle individual model failures gracefully

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🚀 **Quick Screening**
:link: screening_usage_doc
:link-type: doc
Fast model screening with automatic selection
:::

:::{grid-item-card} 🎯 **Model Definitions**
:link: definitions
:link-type: doc  
Pre-configured ML models and parameter grids
:::

:::{grid-item-card} 🔍 **Grid Search**
:link: grid_search
:link-type: doc
Advanced hyperparameter optimization strategies
:::

:::{grid-item-card} 📊 **Evaluation & Visualization**
:link: evaluation
:link-type: doc
Comprehensive model assessment and plotting
:::

:::{grid-item-card} 🔧 **Data Preprocessing**
:link: preprocessing 
:link-type: doc
Data preparation, scaling, and splitting utilities
:::

:::{grid-item-card} ⚡ **Performance Tips**
:link: #performance-optimization
Best practices for large-scale screening
:::
::::

## Quick Start

### Simple Model Evaluation

```python
from polyglotmol.data import MolecularDataset
from polyglotmol.models.api import simple_evaluate

# Load your dataset
dataset = MolecularDataset.from_csv("molecules.csv", 
                                   input_column="SMILES", 
                                   label_columns=["activity"])

# Quick evaluation with default settings
results = simple_evaluate(dataset, target_column="activity")

print(f"Best model: {results['best_model']['model_name']}")
print(f"Best representation: {results['best_model']['representation_name']}")
print(f"Performance: R² = {results['best_model']['metrics']['r2']:.3f}")
```

### Automated Model Screening

```python
from polyglotmol.models.api import quick_screen, thorough_screen

# Fast screening (~5 minutes)
results = quick_screen(dataset, target_column="activity")

# Comprehensive screening (~30 minutes)  
results = thorough_screen(dataset, target_column="activity")

# Access best model
best_estimator = results['best_estimator']
predictions = best_estimator.predict(new_molecules)
```

### Advanced Configuration

```python
from polyglotmol.models import screen_models

# Custom screening with specific settings
results = screen_models(
    dataset=dataset,
    target_column="activity", 
    task_type="regression",
    
    # Representation selection
    representations=["morgan_fp_r2_1024", "rdkit_descriptors", "maccs_keys"],
    
    # Model selection  
    model_corpus="accurate",  # Options: fast, accurate, interpretable, robust
    
    # Multi-stage settings
    stage1_top_percent=0.2,      # Keep top 20% for stage 2
    stage2_top_n=5,              # Report top 5 models
    enable_stage3=True,          # Enable adaptive refinement
    
    # Performance settings
    n_jobs=-1,                   # Use all CPUs
    enable_checkpointing=True    # Save progress
)

print(f"Screening completed: {results['success']}")
print(f"Project saved to: {results['project_dir']}")
```

## Key Features

### Automated Intelligence
- **Multi-Stage Screening**: Coarse → Fine → Adaptive optimization
- **Representation Diversity**: Avoid overfitting to single representation type
- **Smart Parameter Selection**: Scientifically-informed parameter ranges

### High Performance  
- **Parallel Execution**: Multi-core CPU and HPC cluster support
- **Checkpointing**: Resume interrupted screenings
- **Memory Efficiency**: Handle large datasets without memory issues

### User-Friendly Results
- **Publication Plots**: Automatically generated performance visualizations  
- **Model Comparison**: Side-by-side comparison of different approaches
- **Interpretable Reports**: Clear summaries with statistical significance

### Production Ready
- **Model Persistence**: Save and load trained models
- **Reproducible Results**: Fixed random seeds and deterministic outputs
- **Error Handling**: Graceful handling of individual model failures

## Performance Optimization

### Hardware Utilization

```python
# Optimize for your hardware
results = screen_models(
    dataset=dataset,
    target_column="activity",
    
    # CPU optimization
    n_jobs=-2,              # Leave 2 CPUs free for system
    
    # Memory management  
    max_memory_gb=16,       # Limit memory usage
    batch_size=1000,        # Process in batches
    
    # Storage optimization
    save_models="best_only", # Save only top models
    compress_results=True    # Compress checkpoint files
)
```

### HPC Cluster Integration

```python
# SLURM/PBS cluster configuration
from polyglotmol.models import ScreeningPipeline

pipeline = ScreeningPipeline(
    task_type='regression',
    cluster_config={
        'scheduler': 'slurm',
        'nodes': 4,
        'cpus_per_node': 32,
        'memory_per_node': '128GB',
        'time_limit': '24:00:00'
    }
)

# Submit to cluster
job_id = pipeline.submit_screening(dataset, target_column="activity")
```

### Large Dataset Strategies

```python
# For datasets > 100K molecules
def large_dataset_screening(dataset, target_column):
    """Optimized screening for large datasets"""
    
    # 1. Use fast representations first
    fast_results = quick_screen(
        dataset, target_column,
        representations=["morgan_fp_r2_1024", "maccs_keys"]  # Fast to compute
    )
    
    # 2. Select best representation families
    best_repr_types = analyze_representation_performance(fast_results)
    
    # 3. Detailed screening on selected types only
    detailed_results = thorough_screen(
        dataset, target_column,
        representations=best_repr_types,
        enable_stage3=True
    )
    
    return detailed_results

# Memory-efficient processing
def memory_efficient_screening(large_dataset):
    """Handle datasets that don't fit in memory"""
    
    # Split dataset into chunks
    chunk_results = []
    for chunk in large_dataset.iter_chunks(chunk_size=10000):
        chunk_result = quick_screen(chunk, target_column="activity")
        chunk_results.append(chunk_result)
    
    # Aggregate results
    final_result = aggregate_screening_results(chunk_results)
    return final_result
```

### Performance Tips

```{admonition} Best Practices
:class: tip

1. **Start Small**: Use `quick_screen()` on a subset to identify promising approaches
2. **Representation Selection**: Choose 3-5 diverse representations rather than testing all
3. **Progressive Screening**: Stage 1 → evaluate → Stage 2 on best → Stage 3 if needed
4. **Monitor Resources**: Use `htop`/`nvidia-smi` to monitor CPU/GPU usage
5. **Checkpointing**: Always enable checkpointing for long runs
6. **Storage Management**: Clean up intermediate files regularly
```

## Module Contents

```{toctree}
:maxdepth: 1
:hidden:

definitions
preprocessing
grid_search
evaluation
screening_usage_doc
```

## API Reference

For detailed API documentation, see the {doc}`API Models section </api/models/index>`.