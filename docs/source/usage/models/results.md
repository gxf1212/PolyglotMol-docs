# Working with Results

Learn how to access, analyze, and export screening results from MolBlender's SQLite database storage.

## Result Storage

MolBlender uses **SQLite databases** as the primary storage format for screening results, replacing older JSON-only approaches.

### Benefits of SQLite Storage

✅ **Incremental Saving** - Results saved after each model completes
✅ **Crash Recovery** - Resume interrupted screenings automatically
✅ **Efficient Queries** - Fast filtering and aggregation
✅ **Caching** - Skip already-completed combinations
✅ **Portable** - Single `.db` file contains all results

### Enabling Database Storage

```python
from molblender.models.api import universal_screen

results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True,  # Enable SQLite storage
    db_path="./my_screening.db"  # Optional: custom path
)
```

## Database Schema

MolBlender creates three tables to store screening results:

### 1. screening_sessions

Stores metadata about each screening run:

```sql
CREATE TABLE screening_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT,
    task_type TEXT,  -- 'regression' or 'classification'
    primary_metric TEXT,  -- 'pearson_r', 'f1', etc.
    dataset_name TEXT,
    cv_folds INTEGER,
    test_size REAL,
    random_state INTEGER,
    n_models_evaluated INTEGER,
    n_representations INTEGER,
    best_score REAL,
    mean_score REAL,
    std_score REAL
)
```

### 2. model_results

Individual model evaluation results:

```sql
CREATE TABLE model_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,  -- Links to screening_sessions
    model_name TEXT NOT NULL,
    representation_name TEXT NOT NULL,
    representation_config TEXT NOT NULL,  -- JSON representation config
    model_config TEXT NOT NULL,  -- JSON full model config
    primary_metric REAL,  -- Test-set metric value (for display, not for selection)
    primary_metric_name TEXT,
    rank INTEGER,
    hpo_score REAL,  -- HPO CV score when available
    cv_fold_scores TEXT,  -- JSON array of CV fold scores
    training_time REAL DEFAULT 0.0,
    n_features INTEGER DEFAULT 0,
    model_params TEXT,  -- JSON model configuration
    predictions TEXT,  -- JSON predictions
    feature_importance TEXT,  -- JSON feature importance
    model_artifact BLOB,  -- Serialized model (if configured)
    stage INTEGER DEFAULT 1,  -- 1=defaults, 2=coarse/customized, 3=fine, 4=ultrafine
    hpo_stage TEXT,  -- NULL, 'coarse', 'customized', 'fine', or 'ultrafine'
    hpo_method TEXT,  -- 'grid' or 'optuna'
    best_params TEXT,  -- JSON best params from HPO
    all_metrics TEXT,  -- JSON dict of all computed metrics
    grid_search_results TEXT,  -- JSON grid search detail (HPO)
    train_indices TEXT,  -- JSON training indices
    test_indices TEXT,  -- JSON test indices
    val_indices TEXT,  -- JSON validation indices (holdout mode)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES screening_sessions (session_id)
)
```

The `stage` and `hpo_stage` columns follow a four-stage HPO pipeline: **stage 1** stores default-parameter results, **stage 2** stores coarse and customized HPO results, **stage 3** stores fine-tuned results, and **stage 4** stores ultrafine results. The `primary_metric` column holds the test-set score for display purposes only; **model selection uses CV or HPO scores**, not this value (see {doc}`methodology`).

### 3. dataset_info

Train/test split information:

```sql
CREATE TABLE dataset_info (
    session_id TEXT PRIMARY KEY,
    target_column TEXT,
    dataset_n_train_samples INTEGER,
    dataset_n_test_samples INTEGER,
    train_true_values TEXT,  -- JSON array
    test_true_values TEXT,   -- JSON array
    train_input_data TEXT,   -- JSON SMILES/identifiers
    test_input_data TEXT     -- JSON SMILES/identifiers
)
```

## Accessing Results from Python

### Quick Access from Return Value

```python
results = universal_screen(dataset, target_column="activity")

# Best model information
best = results['best_model']
print(f"Model: {best['model_name']}")
print(f"Representation: {best['representation_name']}")
print(f"Primary metric: {best['primary_metric']:.3f}")

# Access all metrics (r2, rmse, mae, pearson_r, etc.)
print(f"R²: {best['all_metrics']['r2']:.3f}")

# All results sorted by performance
for result in results['top_models'][:5]:  # Top 5
    print(f"{result['model_name']} + {result['representation_name']}: {result['primary_metric']:.3f}")
```

### Loading from Database

```python
from molblender.models.api.utils import ScreeningResultsDB

# Connect to the database
db = ScreeningResultsDB("./my_screening.db")

# Load the latest session's results
results = db.load_comprehensive_results()

# Load combined results from all sessions
results = db.load_comprehensive_results(use_all_sessions=True)

# Access results
print(f"Best score: {results['summary']['best_score']}")
for model in results['model_results']:
    print(f"{model['model_name']}: {model['primary_metric']}")
```

### Direct SQL Queries

For advanced analysis, query the database directly:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("./my_screening.db")

# Get all results sorted by primary metric
df = pd.read_sql_query("""
    SELECT
        model_name,
        representation_name,
        primary_metric,
        training_time,
        n_features
    FROM model_results
    WHERE session_id = ?
    ORDER BY primary_metric DESC
""", conn, params=("screening_20250115_103045",))

# Get summary statistics by model type
summary = pd.read_sql_query("""
    SELECT
        model_name,
        COUNT(*) as n_representations,
        AVG(primary_metric) as mean_score,
        MAX(primary_metric) as best_score,
        AVG(training_time) as avg_time
    FROM model_results
    WHERE session_id = ?
    GROUP BY model_name
    ORDER BY mean_score DESC
""", conn, params=("screening_20250115_103045",))

conn.close()
```

## Exporting Results

### Export to JSON

```python
from molblender.models.api.utils import ScreeningResultsDB

db = ScreeningResultsDB("./my_screening.db")

# Export a session's results to JSON
db.export_to_json(
    session_id="screening_20250115_103045",
    output_path="./results.json",
)
```

### Export to CSV

```python
from molblender.models.api.utils import export_results_csv

# Export results table to CSV
export_results_csv(
    results=results,
    output_path="./screening_results.csv"
)

# Or use pandas for custom exports
import sqlite3
import pandas as pd

conn = sqlite3.connect("./my_screening.db")
df = pd.read_sql_query("SELECT * FROM model_results", conn)
df.to_csv("./all_results.csv", index=False)
conn.close()
```

### Save Best Model

```python
# The best_model entry in the results dict contains metadata, not the trained
# estimator object. To save and reuse a trained model, retrain it using the
# stored configuration and parameters.
best = results['best_model']
model_params = best.get('model_params', {})

# Reconstruct and train the model for deployment
from molblender.models.api.screening_engine.model_registry import get_model_registry

registry = get_model_registry()
config = registry.get_model_config(best['model_name'])
estimator = config.create_estimator(**model_params)
estimator.fit(X_train, y_train)

import joblib
joblib.dump(estimator, './best_model.pkl')
```

### Programmatic Model Recreation

For full automation, `recreate_model_from_config` runs the entire retrain pipeline in one call. First retrieve the split indices from the database, then pass them to the recreation function:

```python
import json
import sqlite3
from molblender.models.api.export import recreate_model_from_config

# Load split indices from database
conn = sqlite3.connect("./my_screening.db")
row = conn.execute(
    "SELECT train_indices, test_indices FROM model_results ORDER BY primary_metric DESC LIMIT 1"
).fetchone()
conn.close()

recreation = recreate_model_from_config(
    representation=row["representation_name"],
    model_name=row["model_name"],
    hyperparameters=json.loads(row["model_params"]),
    dataset_path="data.csv",
    target_column="pIC50",
    train_indices=json.loads(row["train_indices"]),
    test_indices=json.loads(row["test_indices"]),
    output_dir="./retrained/"
)

print(f"Test R²: {recreation['test_metrics']['r2']:.4f}")
```

The function returns:

| Key | Description |
|-----|-------------|
| `model` | Trained estimator object |
| `train_predictions`, `test_predictions` | Predictions for each set |
| `train_metrics`, `test_metrics` | `pearson_r`, `r2`, `mae`, `rmse` |
| `X_train`, `X_test`, `y_train`, `y_test` | Feature matrices and labels |
| `configuration` | Representation, model name, hyperparameters |

To generate standalone Python code (instead of calling the function directly):

```python
from molblender.models.api.export import generate_reproduction_code

code = generate_reproduction_code(
    representation="morgan_fp_r2_1024",
    model_name="xgboost",
    hyperparameters=best_model["model_params"],
    dataset_path="data.csv",
    target_column="pIC50",
    train_indices=train_idx,
    test_indices=test_idx
)

with open("reproduce.py", "w") as f:
    f.write(code)
```

## Result Structure

### Complete Result Dictionary

```python
{
    'success': True,
    'timestamp': '2025-01-15T10:30:45',

    # Selection metadata
    'selection': {
        'source': 'cv_score',  # 'cv_score' or 'hpo_score'
        'test_primary_metric_used_for_selection': False,
        'excluded_incomparable_count': 0
    },

    # Best model (selected by CV/HPO score, not test score)
    'best_model': {
        'model_name': 'xgboost',
        'representation_name': 'morgan_fp_r2_1024',
        'task_type': 'regression',
        'primary_metric': 0.852,  # Test-set score for display
        'primary_metric_name': 'pearson_r',
        'all_metrics': {
            'r2': 0.852,
            'rmse': 0.543,
            'mae': 0.421,
            'pearson_r': 0.924,
            'spearman_rho': 0.912,
            'kendall_tau': 0.765
        },
        'cv_scores': [0.831, 0.867, 0.849, 0.856, 0.857],
        'cv_fold_scores': [0.831, 0.867, 0.849, 0.856, 0.857],
        'cv_mean': 0.852,
        'cv_std': 0.012,
        'training_time': 12.34,
        'prediction_time': 0.45,
        'cohort_fingerprint': 'abc123...',
        'coverage': 1.0,
        'model_params': {'n_estimators': 200, 'learning_rate': 0.1},
        'hpo_stage': 'fine',       # Only present when HPO ran
        'stage': 3,
    },

    # Top models sorted by CV/HPO score
    'top_models': [
        {'model_name': 'xgboost', 'primary_metric': 0.852, ...},
        {'model_name': 'random_forest', 'primary_metric': 0.841, ...},
        ...
    ],
    'ranked_top_models': [...],  # Alias of top_models

    # All results (sorted by selection score)
    'all_results': [
        {'model_name': 'xgboost', 'primary_metric': 0.852, ...},
        ...
    ],
    'detailed_results': [...],  # Alias of all_results

    # Statistical summary
    'summary': {
        'n_models_evaluated': 18,
        'n_representations': 6,
        'n_unique_models': 3,
        'best_score': 0.852,
        'worst_score': 0.512,
        'mean_score': 0.764,
        'std_score': 0.089,
        'median_score': 0.771,
        'mean_cv_score': 0.758,
        'std_cv_score': 0.092
    },

    # Configuration snapshot
    'screening_config': {
        'task_type': 'regression',
        'primary_metric': 'pearson_r',
        'cv_folds': 5,
        'test_size': 0.2,
        'random_state': 42
    },
}
```

**Field Notes:**
- `cv_scores` and `cv_fold_scores` hold the same data — both are aliases for the per-fold CV score list.
- `coverage` indicates what fraction of the original cohort was retained after quality filtering (1.0 = all molecules kept).
- `cohort_fingerprint` identifies the exact subset of molecules evaluated, enabling cross-session comparison.
- `model_params` is the config used to create the estimator. If HPO ran, `best_params` is stored separately in the database.
- `selection['source']` indicates which score the best model was chosen on: `'cv_score'` if only Stage 1 ran, `'hpo_score'` if HPO contributed.
- `selection['test_primary_metric_used_for_selection']` is always `False` — the test set was never used to pick the winner, only to report final metrics.
- `selection['excluded_incomparable_count']` counts models that had no valid selection score and were skipped from ranking.

## Caching and Resuming

### Automatic Caching

With `enable_db_storage=True`, MolBlender automatically:

1. **Checks existing results** before training
2. **Skips completed combinations** (same model + representation)
3. **Saves incrementally** after each successful evaluation
4. **Resumes from interruption** (Ctrl+C, crashes, timeouts)

```python
# First run - trains 20 models
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True,
    db_path="./cache.db"
)
# Training completes 15/20 models before interruption...

# Second run - automatically skips 15 completed models
results = universal_screen(
    dataset=dataset,
    target_column="activity",
    enable_db_storage=True,
    db_path="./cache.db"  # Same database
)
# Only trains remaining 5 models!
```

### Manual Cache Management

```python
from molblender.models.api.utils import get_cache_info, clear_cache

# Inspect the file-based result cache (separate from SQLite DB storage)
cache_info = get_cache_info()
print(f"Cache hits: {cache_info.get('hits')}")
print(f"Cache misses: {cache_info.get('misses')}")

# Clear expired cache entries
clear_cache()
```

## Launching the Dashboard

The easiest way to explore results is through the interactive dashboard:

```bash
# View results from database
molblender view ./my_screening.db

# View results from output directory
molblender view ./screening_results_20250115_103045

# Custom port
molblender view ./my_screening.db --port 8503
```

The dashboard automatically:
- Loads all sessions from the database
- Provides interactive charts and tables
- Supports metric switching
- Enables filtering and sorting
- Allows CSV export of filtered results

See {doc}`../dashboard/index` for complete dashboard documentation.

## Programmatic Analysis

### Compare Multiple Runs

```python
import sqlite3
import pandas as pd

# Load results from multiple databases
dbs = ["run1.db", "run2.db", "run3.db"]
all_results = []

for db_path in dbs:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            model_name,
            representation_name,
            primary_metric,
            training_time
        FROM model_results
    """, conn)
    df['run'] = db_path
    all_results.append(df)
    conn.close()

combined = pd.concat(all_results)
print(combined.groupby('model_name')['primary_metric'].agg(['mean', 'std', 'max']))
```

### Extract Predictions

```python
# Get predictions from best model (only present for best_model, not all_results)
best_model = results['best_model']
predictions = best_model.get('predictions')

if predictions:
    test_pred = predictions['test_predictions']

    # True values are stored in the dataset_info DB table, not in the result dict
    import sqlite3
    conn = sqlite3.connect("./my_screening.db")
    row = conn.execute(
        "SELECT test_true_values FROM dataset_info LIMIT 1"
    ).fetchone()
    import json
    test_true = json.loads(row[0]) if row else None
    conn.close()

    if test_true:
        from sklearn.metrics import r2_score, mean_absolute_error
        print(f"R²: {r2_score(test_true, test_pred):.3f}")
        print(f"MAE: {mean_absolute_error(test_true, test_pred):.3f}")
```

### Feature Importance Analysis

```python
# Extract feature importance from tree-based models
best_model = results['best_model']

# Statistics about feature importance distribution
if best_model.get('has_feature_importance'):
    stats = best_model['feature_importance_stats']
    print(f"Mean importance: {stats['mean']:.6f}")
    print(f"Max importance: {stats['max']:.6f}")
    print(f"Feature count: {stats['n_features']}")

    # Top-N named features (when feature names are available)
    top_named = best_model.get('feature_importance_top_named', [])
    if top_named:
        import pandas as pd
        df = pd.DataFrame(top_named)
        print(df.head(10))
```

## Best Practices

```{admonition} Recommendations
:class: tip

1. **Always enable `enable_db_storage=True`** for runs > 5 minutes
2. **Use descriptive `db_path` names** with timestamps or experiment IDs
3. **Export to JSON** for archiving and sharing
4. **Retrain best models** from stored `model_params` for production deployment — the result dict does not contain the trained estimator
5. **Use the dashboard** for initial exploration before programmatic analysis
6. **Query database directly** for custom statistics and aggregations
```

```{admonition} Common Pitfalls
:class: warning

- **Don't delete `.db` files** until you've exported results
- **Check `success` field** before accessing results
- **Handle missing predictions** - not all models save prediction details
- **Close database connections** when done with direct SQL queries
```

## Troubleshooting

**Q: Database file is locked**
```python
# Close any open connections
conn.close()

# Or use context manager (auto-closes)
with sqlite3.connect("results.db") as conn:
    df = pd.read_sql_query("SELECT * FROM model_results", conn)
```

**Q: Can't find session_id**
```python
# List all sessions
conn = sqlite3.connect("results.db")
sessions = pd.read_sql_query("SELECT session_id, timestamp FROM screening_sessions", conn)
print(sessions)
conn.close()
```

**Q: Results dictionary missing keys**
```python
# Always check success status
if not results.get('success', False):
    print(f"Screening failed: {results.get('error')}")
    exit(1)

# Use .get() with defaults for optional keys
best_score = results.get('summary', {}).get('best_score', float('-inf'))
```

## Next Steps

- **Visualize results**: {doc}`../dashboard/index` - Interactive exploration
- **Run more screenings**: {doc}`screening` - Function reference
- **Understanding models**: {doc}`models` - Model catalog and selection
