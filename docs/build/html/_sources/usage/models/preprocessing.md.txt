# Preprocessing

The preprocessing module provides utilities for data preparation, splitting, scaling, and handling imbalanced datasets. These tools are designed to work seamlessly with both numpy arrays, pandas DataFrames, and PolyglotMol's MolecularDataset class.

## Data Splitting

Split your data into training, validation, and test sets:

```python
from polyglotmol.models.preprocessing import split_dataset
import numpy as np

# Create sample data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, 100)

# Split into train2/val/test sets
splits = split_dataset(
    X=X, 
    y=y,
    test_size=0.2,
    val_size=0.1,
    stratify=True,
    random_state=42
)

# Access the splits
X_train, y_train = splits['X_train'], splits['y_train']
X_val, y_val = splits['X_val'], splits['y_val']
X_test, y_test = splits['X_test'], splits['y_test']

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
# Output: Train: 70, Val: 10, Test: 20
```

For more complex splitting scenarios, you can use the `DataSplitter` class:

```python
from polyglotmol.models.preprocessing import DataSplitter

# Create a splitter
splitter = DataSplitter(
    test_size=0.2,
    val_size=0.1,
    stratify=True,
    random_state=42,
    save_indices=True,
    output_dir='./splits'
)

# Split the data
splits = splitter.split(X, y)

# Save and load split indices for reproducibility
splitter.load_indices('./splits/split_indices_20230101-120000.npz')
new_splits = splitter.split_by_indices(X, y)
```

## Feature Scaling

Scale your features for better model performance:

```python
from polyglotmol.models.preprocessing import scale_features
import numpy as np

# Create sample data
X_train = np.random.rand(80, 5) * 100
X_test = np.random.rand(20, 5) * 100

# Scale features using standardization
X_train_scaled, X_test_scaled = scale_features(
    X_train=X_train,
    X_test=X_test,
    scaler_type='standard'
)

print(f"Original range: {X_train.min():.1f}-{X_train.max():.1f}")
# Output: Original range: 0.0-99.9
print(f"Scaled range: {X_train_scaled.min():.1f}-{X_train_scaled.max():.1f}")
# Output: Scaled range: -2.0-2.5

# Scale with MinMax to [0,1] range
X_train_minmax, X_test_minmax = scale_features(
    X_train=X_train,
    X_test=X_test,
    scaler_type='minmax'
)

print(f"MinMax scaled range: {X_train_minmax.min():.1f}-{X_train_minmax.max():.1f}")
# Output: MinMax scaled range: 0.0-1.0
```

You can also scale only specific columns in a DataFrame:

```python
import pandas as pd

# Create a DataFrame with mixed feature types
df_train = pd.DataFrame({
    'numeric1': np.random.rand(80) * 100,
    'numeric2': np.random.rand(80) * 50,
    'category': np.random.choice(['A', 'B', 'C'], 80)
})

df_test = pd.DataFrame({
    'numeric1': np.random.rand(20) * 100,
    'numeric2': np.random.rand(20) * 50,
    'category': np.random.choice(['A', 'B', 'C'], 20)
})

# Scale only numeric columns
df_train_scaled, df_test_scaled = scale_features(
    X_train=df_train,
    X_test=df_test,
    scaler_type='standard',
    columns=['numeric1', 'numeric2']
)

print(df_train_scaled.head(2))
# Output:
#    numeric1  numeric2 category
# 0  0.123456  0.789012        A
# 1 -0.567890  1.234567        B
```

## Feature Normalization

Normalize samples to unit norm:

```python
from polyglotmol.models.preprocessing import normalize_features
import numpy as np

# Create sample data
X = np.array([[1, 2, 3], [4, 5, 6]])

# Normalize to unit L2 norm
X_norm = normalize_features(X, norm='l2')

# Verify normalization
norms = np.sqrt(np.sum(X_norm**2, axis=1))
print(f"Norms: {norms}")
# Output: Norms: [1. 1.]
```

## Handling Imbalanced Data

Deal with class imbalance for classification tasks:

```python
from polyglotmol.models.preprocessing import handle_imbalanced_data
import numpy as np

# Create imbalanced dataset
X = np.random.rand(100, 5)
y = np.array([0] * 90 + [1] * 10)  # 90% class 0, 10% class 1

print(f"Original class distribution: {np.bincount(y)}")
# Output: Original class distribution: [90 10]

# Balance using SMOTE
X_resampled, y_resampled = handle_imbalanced_data(
    X=X,
    y=y,
    method='smote',
    random_state=42
)

print(f"Resampled class distribution: {np.bincount(y_resampled)}")
# Output: Resampled class distribution: [90 90]

# Other methods
X_under, y_under = handle_imbalanced_data(X, y, method='random_undersampling')
print(f"Undersampled class distribution: {np.bincount(y_under)}")
# Output: Undersampled class distribution: [10 10]

X_over, y_over = handle_imbalanced_data(X, y, method='random_oversampling')
print(f"Oversampled class distribution: {np.bincount(y_over)}")
# Output: Oversampled class distribution: [90 90]
```

## Preparing Data from MolecularDataset

Work directly with PolyglotMol's MolecularDataset:

```python
from polyglotmol.data import MolecularDataset
from polyglotmol.models.preprocessing import prepare_data_for_modeling

# Create a dataset
dataset = MolecularDataset.from_csv(
    filepath="activity_data.csv",
    input_column="SMILES",
    label_columns=["activity"]
)

# Generate features
dataset.add_features("morgan_fp_r2_1024")

# Prepare data for modeling
data = prepare_data_for_modeling(
    dataset=dataset,
    feature_column="morgan_fp_r2_1024",
    target_column="activity",
    test_size=0.2,
    preprocessing='standard',
    stratify=False,
    random_state=42
)

# Access prepared data
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
# Output: Train samples: 80, Test samples: 20
```

## Outlier Detection

Identify outliers in your dataset:

```python
from polyglotmol.models.preprocessing import detect_outliers
import numpy as np

# Create data with outliers
X = np.random.normal(0, 1, (100, 5))
X[0] = [10, 10, 10, 10, 10]  # Add an outlier

# Detect outliers using IQR
outliers_iqr = detect_outliers(X, method='iqr', threshold=1.5)
print(f"Number of outliers (IQR): {np.sum(outliers_iqr)}")
# Output: Number of outliers (IQR): 1

# Detect outliers using Z-score
outliers_zscore = detect_outliers(X, method='zscore', threshold=3.0)
print(f"Number of outliers (Z-score): {np.sum(outliers_zscore)}")
# Output: Number of outliers (Z-score): 1
```

## Feature Statistics

Get summary statistics for your features:

```python
from polyglotmol.models.preprocessing import get_feature_stats
import numpy as np

# Create sample data
X = np.random.rand(100, 5)

# Get statistics
stats = get_feature_stats(X)

# Print statistics for the first feature
print(f"Feature 0 stats:")
print(f"  Min: {stats['min'][0]:.4f}")
print(f"  Max: {stats['max'][0]:.4f}")
print(f"  Mean: {stats['mean'][0]:.4f}")
print(f"  Std: {stats['std'][0]:.4f}")
print(f"  Median: {stats['median'][0]:.4f}")
# Output:
# Feature 0 stats:
#   Min: 0.0012
#   Max: 0.9932
#   Mean: 0.4876
#   Std: 0.2901
#   Median: 0.4892
```

## API Reference

For detailed API documentation, see {doc}`/api/models/preprocessing`.