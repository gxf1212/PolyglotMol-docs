# Definition

The model definition module provides a collection of pre-configured machine learning models and parameter grids for regression and classification tasks. It also includes utilities for model registration, instantiation, and configuration management.

## Model Registry

PolyglotMol maintains a registry of models and their configurations to make it easy to access and use different models consistently.

```python
from polyglotmol.models import get_regressor_corpus, get_classifier_corpus

# Get all available regression models
regressors = get_regressor_corpus()
print(f"Available regression models: {list(regressors.keys())}")
# Output: Available regression models: ['Linear Regression', 'Ridge Regression', 'Lasso Regression', ...]

# Get a subset of classification models
classifiers = get_classifier_corpus(['Random Forest', 'SVC', 'Logistic Regression'])
print(f"Selected classifiers: {list(classifiers.keys())}")
# Output: Selected classifiers: ['Random Forest', 'SVC', 'Logistic Regression']
```

## Available Models

### Regression Models

The package provides the following regression models with default configurations:

```python
from polyglotmol.models import list_available_models

regression_models = list_available_models('regression')
print(regression_models)
# Output: ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 
#          'Bayesian Ridge', 'Huber Regression', 'Theil-Sen Regression', 
#          'Passive Aggressive Regression', 'SVR', 'ARD Regression', 
#          'K-Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Extra Trees',
#          'Gradient Boosting', 'AdaBoost', 'MLP', 'XGBoost', 'LightGBM', 'CatBoost']
```

### Classification Models

The package provides the following classification models with default configurations:

```python
classification_models = list_available_models('classification')
print(classification_models)
# Output: ['Logistic Regression', 'SGD Classifier', 'SVC', 'K-Nearest Neighbors',
#          'Decision Tree', 'Random Forest', 'Extra Trees', 'Gradient Boosting',
#          'AdaBoost', 'MLP', 'XGBoost', 'LightGBM', 'CatBoost']
```

## Parameter Grids

Each model comes with a predefined parameter grid for hyperparameter optimization:

```python
from polyglotmol.models import get_regression_param_grids, get_classification_param_grids

# Get parameter grid for Random Forest regression
rf_params = get_regression_param_grids(['Random Forest'])['Random Forest']
print(rf_params)
# Output: {'n_estimators': [50, 100, 200],
#          'max_depth': [None, 10, 20],
#          'min_samples_split': [2, 5, 10],
#          'min_samples_leaf': [1, 2, 4],
#          'bootstrap': [True, False]}

# Customize parameter grid for SVC classification
custom_svc_grid = {'SVC': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}}
svc_params = get_classification_param_grids(['SVC'], custom_grids=custom_svc_grid)
print(svc_params['SVC'])
# Output: {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
```

## Instantiating Models

You can instantiate models from the registry with custom parameters:

```python
from polyglotmol.models import instantiate_model

# Create a Random Forest regressor with custom parameters
rf_model = instantiate_model('regression', 'Random Forest', n_estimators=200, max_depth=10)
print(type(rf_model).__name__, rf_model.get_params())
# Output: RandomForestRegressor {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 
#                              'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, ...}

# Create a Logistic Regression classifier
log_reg = instantiate_model('classification', 'Logistic Regression', C=0.5, max_iter=2000)
print(type(log_reg).__name__, log_reg.get_params())
# Output: LogisticRegression {'C': 0.5, 'class_weight': None, 'dual': False, 
#                           'fit_intercept': True, 'intercept_scaling': 1, ...}
```

## Saving and Loading Model Configurations

You can save and load model configurations to ensure reproducibility:

```python
from polyglotmol.models import save_model_config, load_model_config, recreate_model_from_config
from sklearn.ensemble import RandomForestRegressor
import os

# Create and train2 a model
model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

# Save configuration
config_path = save_model_config(
    model=model,
    model_name='RF-100',
    model_type='regression',
    path='./models',
    metadata={'description': 'Random Forest for activity prediction'}
)
print(f"Saved configuration to {config_path}")
# Output: Saved configuration to ./models/RF-100_regression_config.json

# Load configuration
config = load_model_config(config_path)
print(f"Model class: {config['model_class']}, parameters: {config['parameters']}")
# Output: Model class: RandomForestRegressor, parameters: {'n_estimators': 100, 'random_state': 42, ...}

# Recreate model from configuration
recreated_model = recreate_model_from_config(config)
print(type(recreated_model).__name__)
# Output: RandomForestRegressor
```

## Custom Models

You can register your own custom models to the registry:

```python
from polyglotmol.models import register_model
from sklearn.ensemble import GradientBoostingRegressor

# Register a custom model
register_model(
    task_type='regression',
    name='Custom GBM',
    model_class=GradientBoostingRegressor,
    default_params={'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5},
    param_grid={'n_estimators': [50, 150, 250], 'learning_rate': [0.01, 0.05, 0.1]}
)

# Use the custom model
custom_model = instantiate_model('regression', 'Custom GBM')
print(type(custom_model).__name__, custom_model.get_params())
# Output: GradientBoostingRegressor {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5, ...}
```

## API Reference

For detailed API documentation, see {doc}`/api/models/definitions`.