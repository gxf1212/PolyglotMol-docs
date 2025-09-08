# Multi-Stage Model Screening

PolyglotMol provides an intelligent multi-stage screening system that automatically finds the best model and molecular representation combination for your data.

## Overview

The screening process uses a 3-stage approach:

1. **Stage 1 - Coarse Screening**: Quick evaluation with minimal hyperparameters (3^n combinations)
2. **Stage 2 - Fine Screening**: Detailed optimization on top performers  
3. **Stage 3 - Adaptive Refinement** (optional): Smart parameter tuning based on results

```{admonition} Key Benefits
:class: tip
- Automatically tests hundreds of model+representation combinations
- Ensures representation diversity (doesn't just pick one type)
- Parallel execution with checkpointing for fault tolerance
- Saves only the best models to conserve disk space
```

## Quick Start

### Basic Usage

```python
from polyglotmol.models import screen_models

# Screen models on your dataset
results = screen_models(
    dataset=my_dataset,
    target_column='activity',
    task_type='regression'
)

# Access results
print(f"Best model: {results['best_model']['model_name']}")
print(f"Best representation: {results['best_model']['representation_name']}")
print(f"Performance (R²): {results['best_model']['primary_metric']:.4f}")

# Get the trained model
best_estimator = results['best_estimator']
```

### Convenience Functions

```python
from polyglotmol.models import quick_screen, thorough_screen, interpretable_screen

# Quick screening with fast models (~5 min)
results = quick_screen(dataset, 'activity')

# Thorough screening with all stages (~30 min)
results = thorough_screen(dataset, 'activity')

# Use only interpretable models
results = interpretable_screen(dataset, 'activity')
```

## Advanced Configuration

### Model Selection

```python
# Use specific model corpus
results = screen_models(
    dataset=dataset,
    target_column='activity',
    model_corpus='accurate'  # Options: 'fast', 'accurate', 'interpretable', 'robust'
)

# Use specific models
results = screen_models(
    dataset=dataset,
    target_column='activity',
    model_names=['Random Forest', 'XGBoost', 'Ridge Regression']
)
```

### Stage Configuration

```python
results = screen_models(
    dataset=dataset,
    target_column='activity',
    
    # Stage 1 settings
    stage1_top_percent=0.2,      # Keep top 20% for stage 2
    stage1_min_combinations=10,   # At least 10 combinations
    stage1_min_representations=4, # At least 4 different representations
    
    # Stage 2 settings  
    stage2_top_n=5,              # Report top 5 models
    
    # Stage 3 settings
    enable_stage3=True,          # Enable adaptive refinement
    stage3_refinement_points=5   # Points to test in refinement
)
```

### Parallel Execution

```python
results = screen_models(
    dataset=dataset,
    target_column='activity',
    n_jobs=-1,              # Use all CPUs
    enable_checkpointing=True  # Save progress (default)
)
```

## Working with Results

### Result Structure

```python
results = {
    'success': True,
    'best_model': {
        'model_name': 'Random Forest',
        'representation_name': 'morgan_fp_r2_2048',
        'parameters': {'n_estimators': 200, 'max_depth': 20},
        'metrics': {'rmse': 0.23, 'r2': 0.89, 'pearson_r2': 0.91}
    },
    'best_estimator': <trained sklearn model>,
    'top_models': [...],  # List of top N models
    'summary': {...},     # Detailed statistics
    'project_dir': './screening_results/regression_screening_12345'
}
```

### Loading Previous Results

```python
from polyglotmol.models import load_screening_results

# Load from project directory
results = load_screening_results('./screening_results/my_project')

# Access saved models
for model_info in results['saved_models']:
    print(f"Stage {model_info['stage']}: {model_info['filename']}")
```

### Comparing Representations

```python
from polyglotmol.models import compare_representations

# Test all representations with one model
df = compare_representations(
    dataset=dataset,
    target_column='activity',
    model_name='Random Forest'
)

print(df.sort_values('score', ascending=False))
# Output:
#   representation              score    n_evaluations
# 0 morgan_fp_r2_2048          0.891    3
# 1 rdkit_descriptors          0.856    3  
# 2 maccs_keys                 0.823    3
```

### Comparing Models

```python
from polyglotmol.models import compare_models

# Test all models with one representation
df = compare_models(
    dataset=dataset,
    target_column='activity', 
    representation_name='morgan_fp_r2_2048'
)

print(df.sort_values('score', ascending=False))
```

## Using the Pipeline API

For repeated screening tasks:

```python
from polyglotmol.models import ScreeningPipeline

# Create pipeline
pipeline = ScreeningPipeline(task_type='regression')

# Add datasets
pipeline.add_dataset(train_data, 'train', 'activity')
pipeline.add_dataset(external_data, 'external', 'activity')

# Configure
pipeline.add_representations(['morgan_fp', 'descriptors'])
pipeline.set_models(model_corpus='accurate')
pipeline.configure(enable_stage3=True)

# Run on all datasets
all_results = pipeline.run_all()

# Compare across datasets
comparison_df = pipeline.compare_datasets()
```

## Model Corpus Options

| Corpus | Description | Example Models |
|--------|-------------|----------------|
| `fast` | Quick training, good for initial screening | Linear/Ridge Regression, Decision Tree, KNN |
| `accurate` | High performance models | Random Forest, XGBoost, LightGBM |
| `interpretable` | Models with clear feature importance | Linear models, Decision Tree |
| `robust` | Resistant to outliers and noise | Huber Regression, Random Forest |
| `minimal` | One model per type for testing | Ridge, RF, SVM, MLP |

## Performance Tips

1. **Start with quick_screen()** to get initial results fast
2. **Use fewer representations** for initial exploration
3. **Enable stage 3** only when you need the absolute best performance
4. **Set n_jobs=-2** to leave one CPU free for system tasks
5. **Monitor disk usage** - models are saved in the project directory

## Interpreting Stage Results

The screening report shows:
- **Stage 1**: Identified promising model+representation combinations
- **Stage 2**: Refined hyperparameters for top combinations  
- **Stage 3**: Final optimization around best parameters

Example output:
```
Stage 1: Evaluated 450 combinations
- Top representations: morgan_fp (0.85), descriptors (0.82)
- Top models: Random Forest (0.85), XGBoost (0.84)

Stage 2: Refined 45 combinations  
- Best: Random Forest + morgan_fp (0.89)
- Runner-up: XGBoost + descriptors (0.87)

Stage 3: Adaptive refinement
- Final: Random Forest + morgan_fp (0.91)
- Optimal parameters: n_estimators=237, max_depth=18
```

## Troubleshooting

**Out of Memory**: Reduce n_jobs or use model_corpus='fast'

**Screening Takes Too Long**: 
- Disable stage 3
- Use quick_screen() 
- Reduce representations

**Want More Control**: Use MultiStageScreening class directly:
```python
from polyglotmol.models import MultiStageScreening

screening = MultiStageScreening(
    project_name='my_screening',
    task_type='regression'
)
# ... customize further
```

