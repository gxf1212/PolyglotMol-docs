# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Dataset Splitting Documentation** (`docs/source/usage/data/splitting.md`)
  - Comprehensive guide to all 5 supported splitting strategies (train_test, train_val_test, nested_cv, cv_only, user_provided)
  - Detailed implementation references with code locations and line numbers
  - Visual workflow diagrams for each splitting strategy
  - Best practice recommendations based on dataset size and use case
  - Reproducibility guarantees and fixed random seed documentation
  - Cross-validation protocol details with StratifiedKFold for classification
  - Decision tree for choosing the right splitting strategy
  - Performance considerations and memory efficiency comparisons
- **Boltz-2 AI Structure Prediction Integration** (`src/polyglotmol/representations/AI_fold/boltz2/`)
  - Complete module for extracting embeddings from Boltz-2 protein-ligand complex predictions
  - Support for three embedding types: global (29-33 dim), token-level (4-dim pooled), pairwise distance matrices
  - Intelligent caching system to avoid redundant structure predictions (saves hours of GPU time)
  - Isolated conda environment execution via subprocess to prevent dependency conflicts
  - Automatic YAML input generation with short IDs to avoid Boltz-2 truncation errors
  - MSA server integration for protein sequence alignment
  - Structure file parsing (CIF format) with recursive file discovery
  - Geometric feature extraction including COM, radius of gyration, protein-ligand contacts, confidence scores (pLDDT)
  - Complete test suite with unit tests and integration tests
  - **Note**: Structure prediction via subprocess currently has limitations; users can provide pre-computed CIF files for embedding extraction

- **Protein-Ligand Dataset Support** (`src/polyglotmol/data/`)
  - Extended `Molecule` class with `protein_sequence` and `protein_pdb_path` attributes for protein-ligand complexes
  - New parameters in `MolecularDataset.from_csv()`: `protein_sequence_column` and `protein_pdb_column`
  - Automatic storage of protein information in molecule properties for CSV round-tripping
  - Seamless integration with existing featurization pipeline via kwargs passing
- **Dynamic Metric Resolution System** (`src/polyglotmol/dashboard/metrics/central.py`)
  - Single source of truth for all metric definitions and display names
  - Automatic resolution of `primary_metric` to actual metric names (e.g., "Pearson R²", "MAE")
  - Consistent metric naming across all dashboard components
  - Eliminated hardcoded "Primary Metric" references throughout the interface
  - Enhanced `format_metric_name()` utility with DataFrame context for proper resolution

- **Professional Chart Styling Framework** (`src/polyglotmol/dashboard/components/utils/chart_fonts.py`)
  - Unified font sizing system across all dashboard visualizations
  - Professional color scheme (light blue #6BAED6, light green #74C476, light orange #FD8D3C)
  - Consistent axis styling and formatting for research-quality charts
  - Global font size control via `AXIS_TITLE_FONT_SIZE`, `TICK_LABEL_FONT_SIZE`, `CHART_TITLE_FONT_SIZE`

- **Interactive Table System** (`src/polyglotmol/dashboard/components/tables.py`)
  - Complete replacement of HTML tables with native `st.dataframe()` components
  - Dynamic metric column names with proper formatting
  - Sorting, filtering, and export capabilities
  - Responsive design with container width optimization

- **Comprehensive Distribution Analysis Charts** (`src/polyglotmol/dashboard/components/distribution/charts.py`)
  - Five chart types: Box Plot, Violin Plot, Histogram, Density Plot, Raincloud Plot
  - Unified chart rendering system with consistent styling
  - Professional axis labeling with dynamic metric names
  - Eliminated "undefined" chart titles across all visualizations

- **Comprehensive Methodology Documentation** (`docs/source/usage/models/methodology.md`)
  - Complete explanation of train/test data splitting strategy (80/20 default with `random_state=42`)
  - Detailed 5-fold cross-validation protocol documentation
  - Model training and evaluation workflow with code references
  - Numerical examples and visual diagrams showing data flow
  - Best practices for choosing `test_size` and `cv_folds` based on dataset size
  - Known limitations and future improvements documented
  - Added cross-references from `screening.md` and navigation in `models/index.md`
  - Added CHANGELOG to documentation TOC

### Fixed
- **Dynamic Metric Resolution** (`src/polyglotmol/dashboard/components/`)
  - Fixed "Primary Metric" displaying in Outlier Details table - now shows actual metric name
  - Fixed "Primary Metric" displaying in Distribution Overview charts - now shows formatted axis titles
  - Eliminated "undefined" chart titles across all dashboard visualizations
  - Fixed NameError with undefined 'df' variable in modality comparison charts
  - Resolved duplicate column names in results tables

- **Modality Filter Functionality** (`src/polyglotmol/dashboard/components/charts/performance.py`)
  - Fixed non-responsive modality filter in Representation Analysis section
  - Added debug output for troubleshooting filter behavior
  - Enhanced modality mapping with proper error handling

- **Cross-Validation Reproducibility** (`src/polyglotmol/models/api/core/evaluation/evaluator.py`)
  - Fixed CV random_state issue: Now uses `KFold(random_state=42)` and `StratifiedKFold(random_state=42)` objects
  - Ensures complete reproducibility of both CV scores and test scores across different runs
  - Automatically selects StratifiedKFold for classification tasks to maintain class balance
  - Updated documentation to reflect this fix in methodology.md

### Changed
- **Dashboard UI Reorganization**: Improved navigation and logical grouping of analysis components
  - Performance vs Training Time chart now colored by modality category (fingerprints, language-model, spatial, image, string) instead of individual models for clearer pattern recognition
  - Simplified Model Analysis from 3 to 2 sub-tabs (Performance Deep Dive and Efficiency Analysis)
  - Moved Multi-Dimensional Model Comparison (radar chart) from Performance Analysis to Detailed Results as new 6th sub-tab
  - Combined Hierarchical Clustering Analysis with Correlation Matrix in Detailed Results → Metric Correlation tab for related statistical analysis
  - Moved Statistical Summary to Detailed Results → Outlier & Distribution tab (displayed at top)
  - Removed duplicate Representation Analysis from Model Analysis → Performance Deep Dive (still available in dedicated Representation Analysis tab)
  - Modality Performance Statistics table uses HTML rendering for compatibility (Streamlit dataframe attempted but reverted due to PyArrow compatibility issues with Pandas 1.5.3)

- **Dashboard Individual Model Inspection Improvements**:
  - Made Predictions vs True Values scatter plot square-shaped (650×650 pixels) for better visual proportions
  - Moved Hyperparameter Analysis from Detailed Results to Individual Model Inspection tab for logical grouping
  - Consolidated Model Parameters into Hyperparameter Analysis tab (reduced from 4 tabs to 3)
  - Enhanced Hyperparameter Analysis now shows: Model/Representation info cards, parameter configuration (cards + table), all performance metrics, and export options (Python dict + JSON)
  - Improved tab structure: Prediction Scatter Plot → Hyperparameter Analysis → Export & Code

- **Dashboard UI Font Hierarchy**:
  - Implemented 3-level tab font size hierarchy for better visual hierarchy and readability
  - Level 1 main tabs: 32px (Overview, Performance Analysis, etc.)
  - Level 2 sub-tabs: 26px (Modality Analysis, Model Analysis, Efficiency Analysis)
  - Level 3 nested tabs: 20px (Modality Overview, Representation Analysis, etc.)

- **Dashboard Performance Analysis Cleanup**:
  - Removed duplicate Efficiency Analysis sub-tab from Model Analysis (now only appears as top-level tab)
  - Streamlined Model Analysis to focus on Category and Specific Model comparisons
  - Removed Statistical Summary section from Comprehensive Distribution Analysis to reduce redundancy

- **Detailed Results Tab Optimization**:
  - Reduced from 6 sub-tabs to 5 by moving Hyperparameter Analysis to Individual Model Inspection
  - Current structure: Results Table, Metric Correlation, Outlier & Distribution, Distribution Analysis, Multi-Dimensional Comparison

## [0.1.0] - 2025-01-XX

### Added
- Initial release of PolyglotMol
- Multi-modal molecular representation generation
- Automated ML model screening system
- Interactive Streamlit dashboard for results visualization
- Support for 60+ molecular representations across multiple modalities (fingerprints, language models, images, spatial)
- Integration with 20+ ML algorithms (scikit-learn, XGBoost, LightGBM, neural networks)
- SQLite-based results persistence with caching support
- Comprehensive evaluation metrics for regression and classification tasks
