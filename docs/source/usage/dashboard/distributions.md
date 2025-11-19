# Distribution Analysis

Deep dive into performance distributions with five chart types, flexible grouping options, and statistical summaries.

## Overview

The Distribution Analysis tab provides advanced visualization of how performance metrics are distributed across your screening results.

**Key Features:**
- **5 Chart Types** - Box, violin, raincloud, histogram, density
- **Flexible Grouping** - By model, representation, modality, or quartile
- **Statistical Summaries** - Quartiles, outliers, distribution shapes
- **Interactive Filtering** - Dynamic subgroup selection

**Best For:** Understanding variance, identifying outliers, comparing distributions, statistical analysis

## Tab Location

**Navigation:** Dashboard → Distribution Analysis (Tab 2)

## Chart Type Selector

Choose from five visualization types, each optimized for different analysis needs.

### Chart Type Comparison

| **Chart Type** | **Best For** | **Shows** | **Speed** |
|---------------|-------------|-----------|-----------|
| **Box Plot** | Quartiles, outliers | Q1, Median, Q3, whiskers | ⚡⚡⚡ Fast |
| **Violin Plot** | Distribution shape | Density + box plot | ⚡⚡ Medium |
| **Raincloud Plot** | Complete view | Density + box + points | ⚡ Slower |
| **Histogram** | Frequency bins | Binned counts | ⚡⚡⚡ Fast |
| **Density Plot** | Smooth distribution | KDE curves | ⚡⚡ Medium |

## Chart Type 1: Box Plot

Classic statistical visualization showing quartiles and outliers.

### Visual Structure

```
               Outliers (•)
                  │
    Upper whisker ┤
                  │
    ╭─────────────╮   Q3 (75th percentile)
    │             │
    │    ─────    │   Median (50th percentile)
    │             │
    ╰─────────────╯   Q1 (25th percentile)
                  │
    Lower whisker ┤
                  │
               Outliers (•)
```

### Components Explained

**Box (IQR):**
- **Top edge** - 75th percentile (Q3)
- **Middle line** - Median (50th percentile)
- **Bottom edge** - 25th percentile (Q1)
- **Height** - Interquartile Range (IQR = Q3 - Q1)

**Whiskers:**
- **Upper whisker** - Q3 + 1.5×IQR (or max value within)
- **Lower whisker** - Q1 - 1.5×IQR (or min value within)

**Outliers (dots):**
- Points beyond whiskers
- Unusually good or poor performers

### Interpretation Guide

**Narrow Box:**
```
 ╭──╮
 │  │  ← Small IQR = Consistent performance
 ╰──╯
```
Most models perform similarly. Reliable results.

**Wide Box:**
```
 ╭────╮
 │    │  ← Large IQR = High variability
 │    │
 ╰────╯
```
Performance varies widely. Model/representation choice matters.

**Skewed Distribution:**
```
 ╭────╮
 │────│  ← Median near Q3
 │    │
 ╰────╯
```
Many models cluster at high performance, few poor performers.

**Symmetric Distribution:**
```
 ╭────╮
 │    │
 │──── │  ← Median centered
 │    │
 ╰────╯
```
Evenly distributed performance.

**Many Outliers Below:**
```
      •
    • •
   •
 ╭────╮
 │────│
 ╰────╯
```
Most models good, but some fail badly. Investigate outliers.

### Use Cases

✅ **Quick statistical summary** - See quartiles at a glance
✅ **Outlier detection** - Identify unusual results
✅ **Comparing groups** - Side-by-side box plots
✅ **Publication figures** - Standard scientific visualization

## Chart Type 2: Violin Plot

Combines box plot with kernel density estimation for complete distribution shape.

### Visual Structure

```
       ╱╲          ← Wider = More models
      ╱  ╲            in this range
     ╱    ╲
    ╱ ╭──╮ ╲      ← Box plot inside
   ╱  │▓▓│  ╲
  ╱   ╰──╯   ╲
 ╱           ╲
╱             ╲
```

### Components

**Density (Violin Shape):**
- **Width** - Frequency of models at that performance level
- **Symmetry** - Mirrored for easier visual comparison

**Inner Box Plot:**
- Same interpretation as standard box plot
- Shows quartiles within density shape

### Interpretation Guide

**Single Peak:**
```
   ╱╲
  ╱  ╲
 ╱    ╲
╱      ╲
```
Unimodal distribution. Most models cluster around one performance level.

**Multiple Peaks (Bimodal):**
```
 ╱╲    ╱╲
╱  ╲  ╱  ╲
```
Bimodal distribution. Two distinct performance clusters (e.g., good and poor model families).

**Flat Top:**
```
 ╭────╮
 │    │
 │    │
╱      ╲
```
Uniform distribution. Models spread evenly across performance range.

**Long Tail:**
```
     ╱╲
    ╱  ╲───────
   ╱
  ╱
```
Heavy tail. Few extreme outliers extend distribution.

### Use Cases

✅ **Distribution shape** - Identify multimodal patterns
✅ **Density comparison** - See where models concentrate
✅ **Combined statistics** - Density + quartiles in one view
✅ **Group comparison** - Compare distribution shapes across groups

## Chart Type 3: Raincloud Plot

Most comprehensive visualization combining density, box plot, and individual data points.

### Visual Structure

```
Density  →   ╱╲
Curve       ╱  ╲─────────
           ╱

Box Plot →  ╭────╮
            │──  │
            ╰────╯

Points →    • • •  • •
            •  ••• •
```

### Components

**Top: Density Curve (Half Violin)**
- Smooth distribution shape
- Shows concentration of results

**Middle: Box Plot**
- Quartiles and whiskers
- Statistical summary

**Bottom: Individual Points**
- Each dot = one model-representation combination
- Jittered for visibility
- Exact values visible

### Interpretation Guide

**Density + Points Agreement:**
```
Dense curve matches point clusters
→ Distribution is real, not artifact
```

**Outlier Visibility:**
```
Points far from box
→ Specific models to investigate
```

**Cluster Identification:**
```
Distinct point groups visible
→ Natural performance tiers
```

### Use Cases

✅ **Complete picture** - All information in one chart
✅ **Outlier investigation** - See exact outlier points
✅ **Small datasets** - Individual points meaningful
✅ **Presentations** - Most informative single chart

⚠️ **Performance Note:** Slower with 100+ models due to individual point rendering

## Chart Type 4: Histogram

Binned frequency distribution showing counts in performance ranges.

### Visual Structure

```
 12┤        ████
 10┤        ████
  8┤   ████ ████
  6┤   ████ ████ ████
  4┤   ████ ████ ████
  2┤████████ ████ ████ ████
  0┴────────────────────────
   0.5  0.6  0.7  0.8  0.9

X-axis: Performance bins
Y-axis: Count of models
```

### Components

**Bins:**
- X-axis divided into equal-width intervals
- Default: 20 bins (auto-adjusted based on data range)

**Heights:**
- Number of models in each bin
- Taller bars = more common performance level

### Interpretation Guide

**Normal Distribution:**
```
     ████
   ████████
 ██████████████
```
Bell curve. Most models near mean, fewer at extremes.

**Right-Skewed:**
```
 ████████
   ████
     ██
```
Most models perform poorly, few excellent performers.

**Left-Skewed:**
```
     ██
   ████
 ████████
```
Most models perform well, few poor performers.

**Uniform:**
```
 ████ ████ ████
 ████ ████ ████
```
Models evenly distributed across performance range.

**Bimodal:**
```
 ████      ████
 ████      ████
```
Two distinct performance groups.

**Gaps:**
```
 ████      ████
 ████
```
No models in certain ranges. May indicate representation families.

### Use Cases

✅ **Frequency analysis** - How many models in each range?
✅ **Quick overview** - Fastest visualization
✅ **Identifying modes** - Multiple peaks visible
✅ **Large datasets** - Efficient for 100+ models

## Chart Type 5: Density Plot

Smooth continuous distribution using kernel density estimation (KDE).

### Visual Structure

```
      ╱──╲
     ╱    ╲
    ╱      ╲___
   ╱           ╲___
──╱─────────────────╲──
 0.0              1.0

Smooth curve showing probability density
```

### Components

**KDE Curve:**
- Smoothed version of histogram
- Y-axis: Probability density (area under curve = 1)
- X-axis: Performance values

**Bandwidth:**
- Controls smoothness
- Auto-selected for optimal visualization

### Interpretation Guide

**Sharp Peak:**
```
    │
   ╱╲
  ╱  ╲
 ╱    ╲
```
Most models cluster tightly around one value.

**Broad Peak:**
```
  ╭────╮
 ╱      ╲
╱        ╲
```
Models spread across wider performance range.

**Multiple Peaks:**
```
 ╱╲  ╱╲
╱  ╲╱  ╲
```
Distinct performance clusters. Different model families?

**Long Tail:**
```
     ╱╲
    ╱  ╲──────
   ╱
  ╱
```
Most models similar, few extreme cases.

### Use Cases

✅ **Smooth visualization** - Better than histogram for presentations
✅ **Distribution shape** - Clearer than histogram
✅ **Overlay comparisons** - Multiple groups on same plot
✅ **Continuous interpretation** - No artificial binning

## Grouping Options

Organize results by different categories to reveal patterns.

### No Grouping (Overall Distribution)

**Shows:** Single distribution across all results

**Use For:**
- Overall performance assessment
- Identifying global outliers
- Baseline distribution shape

### Group by: Model Type

**Shows:** Separate distributions for each model

**Example:**
```
RandomForest ╭────╮
XGBoost      ╭──╮
Ridge        ╭────────╮
Lasso        ╭──────╮
```

**Insights:**
- Which model types perform best?
- Which are most consistent (narrow distribution)?
- Which have outliers?

**Use Cases:**
✅ Model selection
✅ Identifying robust model families
✅ Comparing algorithm classes

### Group by: Representation

**Shows:** Distributions for each molecular representation

**Example:**
```
morgan_fp_r2_1024  ╭──╮
rdkit_descriptors  ╭────╮
maccs_keys         ╭──────╮
canonical_smiles   ╭────────╮
```

**Insights:**
- Which representations enable good performance?
- Are some representations more variable?
- Do representation families cluster?

**Use Cases:**
✅ Representation selection
✅ Understanding feature importance
✅ Comparing modalities

### Group by: Modality

**Shows:** Distributions grouped by data type (VECTOR, STRING, MATRIX, IMAGE)

**Example:**
```
VECTOR  ╭──╮
STRING  ╭────────╮
MATRIX  ╭────╮
```

**Insights:**
- Which data modality works best for this task?
- Is STRING (Transformers) competitive with VECTOR (traditional ML)?
- Should we focus on one modality?

**Use Cases:**
✅ High-level strategy decisions
✅ Resource allocation (Transformers vs ML)
✅ Identifying modality-task fit

### Group by: Performance Quartile

**Shows:** Distributions for top 25%, second 25%, third 25%, bottom 25%

**Example:**
```
Q4 (Top 25%)     ╭──╮
Q3 (50-75%)      ╭────╮
Q2 (25-50%)      ╭──────╮
Q1 (Bottom 25%)  ╭────────╮
```

**Insights:**
- Characteristics of top performers
- How much worse are bottom quartile results?
- Is there a clear performance tier?

**Use Cases:**
✅ Identifying elite models
✅ Setting performance thresholds
✅ Understanding performance tiers

## Statistical Summaries

Below each chart, the dashboard displays key statistics.

### Summary Statistics Panel

```
┌─────────────────────────────────────┐
│ Distribution Statistics             │
├─────────────────────────────────────┤
│ Count: 18                           │
│ Mean: 0.764 ± 0.089                │
│ Median: 0.782                       │
│ Range: [0.542, 0.856]              │
│ Q1: 0.724, Q3: 0.823               │
│ IQR: 0.099                          │
│ Outliers: 2 (11%)                  │
└─────────────────────────────────────┘
```

**Metrics Explained:**

**Count** - Total number of results in current view
**Mean ± Std** - Average and standard deviation
**Median** - Middle value (50th percentile)
**Range** - [Minimum, Maximum] values
**Q1, Q3** - First and third quartiles
**IQR** - Interquartile range (Q3 - Q1)
**Outliers** - Count and percentage beyond whiskers

### Interpreting Statistics

**Low Standard Deviation (<0.05):**
```
Most models perform similarly
→ Robust signal in data
→ Model choice less critical
```

**High Standard Deviation (>0.15):**
```
Wide performance variation
→ Model/representation selection crucial
→ Some combinations much better
```

**Mean >> Median:**
```
Right-skewed (few excellent performers)
→ Most models mediocre, few stars
→ Find and use the stars
```

**Median >> Mean:**
```
Left-skewed (few poor performers)
→ Most models good, few failures
→ Avoid the failures
```

**High Outlier Count (>10%):**
```
Many extreme values
→ Investigate outliers
→ May reveal data issues
```

## Practical Workflows

### Workflow 1: Understanding Consistency

**Goal:** How consistent are results across models/representations?

1. **Select Box Plot** chart type
2. **Group by: Model Type**
3. **Observe:**
   - Box widths (IQR) - narrow = consistent
   - Number of outliers - few = reliable
   - Median positions - clustered = similar performance

4. **Interpretation:**
   - **Narrow boxes, few outliers** → Consistent results, most models work
   - **Wide boxes, many outliers** → Inconsistent, careful selection needed

### Workflow 2: Identifying Model Clusters

**Goal:** Are there distinct performance tiers?

1. **Select Violin Plot** chart type
2. **No grouping** (overall distribution)
3. **Look for:**
   - Multiple peaks (bimodal/multimodal)
   - Gaps in distribution

4. **If multiple peaks found:**
   - **Group by: Model Type** to see which models in each cluster
   - **Group by: Representation** to see if representation-driven

5. **Action:**
   - Focus on high-performance cluster
   - Avoid low-performance cluster

### Workflow 3: Representation Comparison

**Goal:** Which representation family is best?

1. **Select Box Plot** chart type
2. **Group by: Representation**
3. **Compare:**
   - Median values (which highest?)
   - Box widths (which most consistent?)
   - Outliers (which most reliable?)

4. **Switch to Group by: Modality**
5. **Identify:**
   - Best modality overall
   - Most consistent modality

6. **Decision:**
   - Use best representation from best modality
   - Consider computational cost trade-offs

### Workflow 4: Outlier Investigation

**Goal:** Understand and address outliers

1. **Select Raincloud Plot** chart type
2. **No grouping**
3. **Identify outlier points** (far from box)

4. **Hover over outlier points** to see:
   - Model name
   - Representation name
   - Exact score

5. **Group by: Model Type**
6. **Check if outliers:**
   - Clustered in specific models → Model issue
   - Spread across models → Representation or data issue

7. **Action:**
   - If model-specific: Avoid those models
   - If representation-specific: Investigate representation quality
   - If random: May be normal variance

### Workflow 5: Publication Figure

**Goal:** Create publication-quality distribution plot

1. **Select Violin Plot** for best balance of information and clarity
2. **Group by: Model Type** or **Modality** (depending on message)
3. **Switch to relevant metric** (R², Pearson R, etc.)
4. **Hover over chart** → Click camera icon → Download PNG
5. **Chart features:**
   - Professional color scheme
   - Clear labels
   - Statistical rigor (box plot + density)

## Tips and Best Practices

```{admonition} Distribution Analysis Tips
:class: tip

1. **Start with Box Plot** - Fastest way to see quartiles and outliers
2. **Use Violin for presentations** - Good balance of detail and clarity
3. **Raincloud for deep dives** - When investigating specific results
4. **Group by Modality first** - High-level strategic view
5. **Switch chart types** - Different visualizations reveal different patterns
6. **Check statistics panel** - Numbers complement visuals
```

```{admonition} Common Mistakes
:class: warning

- **Overinterpreting small samples** - <10 models, statistics unreliable
- **Ignoring grouping** - Overall distribution may hide important patterns
- **Not switching metrics** - Distribution shape can change with metric
- **Forgetting outlier context** - Outliers may be valid exceptional results
```

## Chart Type Selection Guide

| **Your Question** | **Best Chart** | **Grouping** |
|------------------|---------------|--------------|
| What's the median performance? | Box Plot | None |
| Are there performance tiers? | Violin Plot | None |
| Which models are outliers? | Raincloud Plot | Model Type |
| Is performance normally distributed? | Histogram | None |
| Which modality is best? | Box Plot | Modality |
| Are representations consistent? | Violin Plot | Representation |
| How many models in top 10%? | Histogram | None |
| What's the smoothed distribution? | Density Plot | None |

## Exporting Distribution Analysis

**Export Charts:**
1. Hover over chart
2. Click camera icon (top right)
3. Saves as high-resolution PNG

**Export Statistics:**
- Statistics panel displayed on screen
- Use Detailed Results tab for full CSV export

## Next Steps

- **Inspect specific models**: {doc}`model_inspection` - Prediction scatter plots
- **Compare with Performance Analysis**: {doc}`performance` - Overall model rankings
- **Get started**: {doc}`quickstart` - Launch the dashboard
