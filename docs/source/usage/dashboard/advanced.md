# Advanced Features

Explore advanced dashboard capabilities including statistical analysis, multi-session comparison, correlation heatmaps, and efficiency analysis.

## Overview

Beyond basic performance visualization, the dashboard provides sophisticated analysis tools for:

- **Statistical Analysis** - Correlation heatmaps with hierarchical clustering
- **Multi-Session Support** - Compare results from different screening runs
- **Modality Analysis** - Deep dive into data type performance
- **Efficiency Analysis** - Training time vs performance trade-offs
- **Custom Filtering** - Complex multi-criteria filtering
- **Batch Export** - Download multiple result sets

## Correlation Heatmaps

Visualize relationships between all performance metrics simultaneously.

### Accessing Correlation Analysis

**Location:** Performance Analysis tab → Statistical Analysis section (scroll down)

### Heatmap Structure

```
         R²   RMSE  MAE   Pear  Spear
    R²  │1.0  -0.98 -0.95  0.99  0.92│
   RMSE │-0.98 1.0   0.97 -0.96 -0.89│
    MAE │-0.95 0.97  1.0  -0.94 -0.87│
  Pear  │0.99 -0.96 -0.94  1.0   0.94│
  Spear │0.92 -0.89 -0.87  0.94  1.0 │

Color scale: -1.0 (blue) → 0 (white) → 1.0 (red)
```

### Components

**Cells:**
- **Diagonal** - Always 1.0 (metric vs itself)
- **Off-diagonal** - Correlation between metrics
- **Color intensity** - Correlation strength

**Color Scheme:**
- **Deep red** - Strong positive correlation (close to 1.0)
- **White** - No correlation (close to 0)
- **Deep blue** - Strong negative correlation (close to -1.0)

**Hierarchical Clustering:**
- Rows and columns reordered
- Similar metrics grouped together
- Dendrogram shows metric relationships

### Interpretation Guide

**High Positive Correlation (r > 0.9, red):**
```
R² and Pearson R: 0.99 (deep red)
→ Nearly equivalent metrics
→ Using both provides little additional information
```

**High Negative Correlation (r < -0.9, blue):**
```
R² and RMSE: -0.98 (deep blue)
→ Inverse relationship (expected)
→ High R² means low RMSE
```

**Moderate Correlation (0.5 < |r| < 0.9, light colors):**
```
Pearson R and Spearman ρ: 0.85 (light red)
→ Related but measure different aspects
→ Both provide complementary information
```

**Low Correlation (|r| < 0.5, near white):**
```
F1 and ROC-AUC: 0.42 (near white)
→ Independent metrics
→ Consider both for complete picture
```

### Metric Clustering Patterns

**Typical Regression Clusters:**

**Cluster 1: Variance Metrics**
- R², Pearson R, Explained Variance
- High inter-correlation (r > 0.95)

**Cluster 2: Error Magnitude Metrics**
- RMSE, MSE, MAE
- High inter-correlation (r > 0.90)

**Cluster 3: Rank Metrics**
- Spearman ρ, Kendall τ
- Moderate correlation with Cluster 1 (r ≈ 0.85)

**Insight:** Choose one metric from each cluster for comprehensive evaluation.

### Use Cases

**Metric Redundancy Check:**
```
If two metrics have r > 0.95:
→ Choose one for reporting
→ Example: R² and Pearson R²
```

**Complementary Metrics:**
```
If two metrics have 0.5 < r < 0.8:
→ Both provide value
→ Example: R² and Spearman ρ
```

**Inconsistent Models:**
```
If model ranks very differently by two metrics:
→ Check their correlation
→ Low correlation → metrics measure different aspects
→ High correlation → investigate model issues
```

## Multi-Session Comparison

Compare results from multiple screening runs to track improvements or investigate experimental variations.

### Loading Multiple Sessions

**From SQLite Database:**

The dashboard automatically loads all sessions from a database file.

**Session Selector:**
```
┌────────────────────────────────────┐
│ Select Session:                    │
│ [▼ screening_20250115_103045]      │
│    screening_20250115_103045       │
│    screening_20250114_151230       │
│    screening_20250113_092145       │
└────────────────────────────────────┘
```

**Session Information:**
Each session displays:
- Timestamp
- Number of models evaluated
- Best score achieved
- Primary metric used

### Comparing Sessions

**Method 1: Side-by-Side Manual Comparison**

1. **Load first session** → Note top performers
2. **Switch to second session** → Compare
3. **Record differences** in separate spreadsheet

**Method 2: Export and Merge**

1. **Export each session to CSV**
2. **Load in Python/R:**
   ```python
   import pandas as pd

   session1 = pd.read_csv('session1.csv')
   session2 = pd.read_csv('session2.csv')

   # Add session identifiers
   session1['session'] = 'baseline'
   session2['session'] = 'optimized'

   # Combine
   combined = pd.concat([session1, session2])

   # Compare best models
   print(combined.groupby('session')['r2'].max())
   ```

### Common Comparison Scenarios

**Scenario 1: Before/After Optimization**
```
Baseline Run:
- 18 models tested
- Best R²: 0.782
- Mean R²: 0.654

After Feature Engineering:
- 18 models tested (same)
- Best R²: 0.856 (+0.074)
- Mean R²: 0.741 (+0.087)

→ Feature engineering improved performance
```

**Scenario 2: Different Representation Families**
```
Run 1 (Fingerprints only):
- Best: morgan_fp_r2_1024, R² = 0.842

Run 2 (Descriptors only):
- Best: rdkit_descriptors, R² = 0.798

→ Fingerprints better for this dataset
```

**Scenario 3: Model Hyperparameter Tuning**
```
Run 1 (Default parameters):
- XGBoost: R² = 0.831

Run 2 (Tuned parameters):
- XGBoost: R² = 0.856 (+0.025)

→ Tuning provided meaningful improvement
```

## Modality Analysis

Deep dive into performance patterns by data type (VECTOR, STRING, MATRIX, IMAGE).

### Modality Summary Cards

**Location:** Performance Analysis tab → Modality Analysis section

```
┌────────────────────────────────────┐
│ VECTOR Modality                    │
├────────────────────────────────────┤
│ Representations: 4                 │
│ Models Tested: 15                  │
│ Best Score: R² = 0.856             │
│ Mean Score: R² = 0.782             │
│ Best Combination:                  │
│   XGBoost + morgan_fp_r2_1024      │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ STRING Modality                    │
├────────────────────────────────────┤
│ Representations: 1                 │
│ Models Tested: 2                   │
│ Best Score: R² = 0.678             │
│ Mean Score: R² = 0.654             │
│ Best Combination:                  │
│   Transformer + canonical_smiles   │
└────────────────────────────────────┘
```

### Modality Comparison Charts

**Grouped Bar Chart:**
```
         VECTOR    STRING   MATRIX
Best    ████████  ██████   ███████
Mean    ███████   █████    ██████
Worst   █████     ████     █████
```

**Insights:**
- Which modality achieves highest performance?
- Which is most consistent (smallest gap between best and worst)?
- Which provides best cost-benefit?

### Modality-Specific Insights

**VECTOR Analysis:**
- Compare fingerprints vs descriptors
- Identify optimal fingerprint parameters (radius, bits)
- Evaluate language model embeddings

**STRING Analysis:**
- Transformer performance on raw SMILES
- Compare with VECTOR modality
- Assess if Transformer training time worth it

**MATRIX Analysis:**
- CNN performance on molecular matrices
- Compare with flattened VECTOR approach
- Dimensionality effects (32×32 vs 64×64)

**IMAGE Analysis:**
- CNN performance on 2D/3D molecular images
- Image resolution effects
- Rendering method impact

### Cross-Modality Recommendations

**If VECTOR >> STRING:**
```
Best VECTOR: R² = 0.856
Best STRING: R² = 0.678

→ Use traditional ML with fingerprints/descriptors
→ Transformers not worth computational cost
```

**If STRING competitive with VECTOR:**
```
Best VECTOR: R² = 0.842
Best STRING: R² = 0.831

→ Consider Transformers for large datasets
→ Transfer learning may help
```

**If MATRIX >> VECTOR:**
```
Best MATRIX: R² = 0.889
Best VECTOR: R² = 0.782

→ 3D structure information valuable
→ Use CNN for this task
```

## Efficiency Analysis

Balance predictive performance with computational cost.

### Training Time vs Performance Scatter Plot

**Location:** Performance Analysis tab → Efficiency Analysis section

```
R² Score
  0.9 ┤           •RandomForest
      │         • •XGBoost
  0.8 ┤       •     •Ridge
      │     •
  0.7 ┤   •Transformer
      │ •
  0.6 ┤
      └────────────────────
       1s  10s 100s 1000s
          Training Time (log scale)

Ideal models: Top-left corner (high R², low time)
```

### Efficiency Quadrants

**Top-Left (Best):** High performance, fast training
- Example: XGBoost, RandomForest
- **Use for:** Production deployment

**Top-Right:** High performance, slow training
- Example: Transformer, fine-tuned CNN
- **Use for:** When accuracy critical, computational cost acceptable

**Bottom-Left:** Low performance, fast training
- Example: Linear models on poor representations
- **Use for:** Baseline comparisons only

**Bottom-Right (Worst):** Low performance, slow training
- Example: Poorly configured deep learning
- **Avoid:** No benefit

### Efficiency Metrics

**Training Efficiency Ratio:**
$$Efficiency = \frac{R^2}{log_{10}(training\_time\_seconds)}$$

Higher ratio = better efficiency.

**Example:**
```
Model A: R² = 0.85, Time = 10s
  Efficiency = 0.85 / log10(10) = 0.85 / 1.0 = 0.85

Model B: R² = 0.87, Time = 300s
  Efficiency = 0.87 / log10(300) = 0.87 / 2.48 = 0.35

→ Model A more efficient despite lower R²
```

### Use Cases

**Production Deployment:**
```
Requirement: Retrain daily with new data
Constraint: <5 minutes training time

Filter models by time < 300s
→ Select best performer from filtered set
```

**Exploratory Analysis:**
```
No time constraints
→ Use highest R² model regardless of time
```

**HPC Resource Optimization:**
```
Compare: 10 × Ridge (10s each) vs 1 × Transformer (1000s)
Both complete in ~15 minutes
→ Test both approaches, compare best results
```

## Custom Filtering

Advanced multi-criteria filtering for specific use cases.

### Multi-Metric Filtering

**Requirement:** Models that perform well on multiple metrics

```python
# Conceptual example (in future dashboard features)
Filter:
  R² > 0.80 AND
  RMSE < 0.5 AND
  Spearman_rho > 0.85

→ Shows only models meeting all criteria
```

**Current Workaround:**
1. Export to CSV
2. Load in Python/R
3. Apply complex filters:
   ```python
   filtered = df[(df['r2'] > 0.80) &
                 (df['rmse'] < 0.5) &
                 (df['spearman_rho'] > 0.85)]
   ```

### Representation Family Filtering

**Goal:** Compare only within representation families

**Method:**
1. **Use search box** with partial matches:
   - Search "morgan" → All Morgan fingerprints
   - Search "rdkit" → All RDKit representations
   - Search "transformer" → All transformer models

2. **Export filtered results** for each family
3. **Compare best from each family**

### Top-N by Multiple Metrics

**Goal:** Find models in top 10 by both R² and Spearman ρ

**Method:**
```python
# Export to CSV, then:
top_r2 = df.nlargest(10, 'r2')['model_id']
top_spearman = df.nlargest(10, 'spearman_rho')['model_id']

# Intersection
robust_models = set(top_r2) & set(top_spearman)
print(f"Models in top 10 for both: {robust_models}")
```

## Batch Export

Export multiple datasets for offline analysis.

### Export Workflow

**Step 1: Filter in Dashboard**
- Apply performance thresholds
- Use search to select models
- Switch to relevant metric

**Step 2: Export Current View**
- Click "Download CSV" in Detailed Results tab
- Saves filtered results only

**Step 3: Repeat for Different Filters**
- Change filter criteria
- Export again with descriptive filename

**Step 4: Combine Offline**
```python
import pandas as pd
import glob

# Load all exported CSVs
files = glob.glob('export_*.csv')
dfs = [pd.read_csv(f) for f in files]

# Combine
combined = pd.concat(dfs, ignore_index=True)

# Remove duplicates
combined = combined.drop_duplicates(subset=['model_name', 'representation_name'])

# Save master file
combined.to_csv('master_results.csv', index=False)
```

## Performance Optimization for Large Results

When dealing with 1000+ models:

### Caching Strategy

**Automatic Caching:**
- Data loading: 5-minute TTL
- Metric calculations: Cached per metric
- Prediction data: Cached per model

**Manual Cache Clearing:**
- Close and reopen browser
- Or: Dashboard → Settings → Clear Cache (if available)

### Pagination Settings

**Default:** 100 models per page

**Optimization:**
- Use filters to reduce total results
- Search for specific models
- Apply top-N selection

### Memory Management

**If dashboard becomes slow:**
1. **Close other browser tabs**
2. **Clear browser cache** (Ctrl+Shift+Delete)
3. **Apply stricter filters** (show top 50 only)
4. **Restart dashboard** (close terminal, relaunch)

## Tips and Best Practices

```{admonition} Advanced Analysis Tips
:class: tip

1. **Use correlation heatmap** - Identify redundant metrics early
2. **Compare sessions systematically** - Export both, merge in Python
3. **Check efficiency quadrant** - Don't ignore training time
4. **Filter incrementally** - Start broad, narrow progressively
5. **Export at each step** - Save intermediate filtered results
6. **Combine dashboard + code** - Dashboard for exploration, code for detailed analysis
```

```{admonition} Common Advanced Pitfalls
:class: warning

- **Too many metrics displayed** - Use heatmap to select 3-4 key metrics
- **Comparing incompatible sessions** - Ensure same dataset and CV settings
- **Ignoring training time** - Fast models often better for production
- **Not exporting intermediate results** - Dashboard state is temporary
- **Forgetting about modality** - Different modalities need different resources
```

## Next Steps

- **Apply in practice**: {doc}`workflows` - Real-world decision-making
- **Understand metrics better**: {doc}`metrics` - Complete metric guide
- **Basic visualization**: {doc}`performance` - Performance analysis tab
- **Distribution analysis**: {doc}`distributions` - 5 chart types explained
