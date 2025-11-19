# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **DNR Diagnostics Module** (`src/polyglotmol/data/diagnostics/`)
  - Complete implementation of DNR (Different Neighbor Ratio) analysis for dataset quality assessment
  - Paper-accurate parameters from "Upgrading Reliability in Molecular Property Prediction" (similarity threshold 0.5, property diff 1.0 log unit)
  - Activity cliff detection for identifying similar molecules with large property differences
  - Comprehensive visualization suite with 5 plot types (DNR distribution, DNR vs property, activity cliff network, similarity heatmap, neighbor statistics)
  - SVG output format to avoid matplotlib/numpy compatibility issues with PNG
  - One-click full diagnostics workflow with `run_full_diagnostics()` method
  - Automatic sampling for large datasets (>10,000 molecules) to prevent memory issues
  - Integration with MolecularDataset API via `DatasetDiagnostics` class
  - Comprehensive markdown report generation with interpretation guidelines
  - Submodules: `core.py` (main diagnostics), `similarity.py` (fingerprint utilities), `visualization.py` (plotting functions)
  - **Interactive Streamlit Dashboard** (`src/polyglotmol/data/diagnostics/dashboard/`)
    - Integrated into package as proper module
    - Console script entry point: `polyglotmol-diagnostics` command
    - Built-in comprehensive documentation and interpretation guide
    - Adjustable thresholds with real-time recomputation
    - CSV export functionality for results
    - Support for file upload or command-line file path
  - Successfully tested on 120-molecule head.csv dataset with complete output generation
  - Comprehensive module documentation in `src/polyglotmol/data/diagnostics/CLAUDE.md`

- **Protein Data Handling Documentation** (`docs/source/usage/data/protein.md`)
  - Comprehensive guide to Protein class and multi-format support
  - Database retrieval from RCSB PDB, AlphaFold, and UniProt
  - Structure prediction and repair with ESMFold and PDBFixer
  - FASTA, PDB, mmCIF format handling with automatic detection
  - BioPython integration for structural analysis
  - Intelligent caching system for downloads and predictions
  - Protein-ligand dataset integration examples
  - Multi-chain structure support and metadata extraction
  - Sequence validation and cleaning utilities
  - Best practices for high-throughput protein processing

- **Scaffold-Based Splitting Implementation** (`src/polyglotmol/models/api/core/splitting/scaffold.py`)
  - Complete scaffold-based train/test splitting for drug discovery applications
  - Two scaffold generation methods: Bemis-Murcko and Generic (topology-only)
  - Two split strategies: Balanced (greedy size matching) and Random (random assignment)
  - RDKit integration for scaffold computation with error handling
  - Scaffold leakage detection to ensure train/test set separation
  - Integration with `DataSplitter` class via `strategy="scaffold"` parameter
  - Comprehensive unit tests with 18 test cases covering all functionality
  - Documentation added to `docs/source/usage/data/splitting.md` with examples and use cases

- **DNR-Based Splitting Strategy** (`src/polyglotmol/models/api/core/splitting/dnr.py`)
  - Systematically tests model performance on rough SAR regions and challenging molecules
  - Three split modes:
    - **Threshold mode**: Split by DNR threshold (high vs low DNR molecules)
    - **Quantile mode**: Split by DNR quantiles (top X% vs rest)
    - **Neighbor mode**: Split by neighbor presence (isolated vs connected molecules)
  - Configurable parameters: DNR threshold, similarity threshold, property difference threshold
  - Enables systematic evaluation of two major error modes from "Upgrading Reliability in Molecular Property Prediction" paper
  - Integration with `DataSplitter` class via `strategy="dnr"` parameter
  - Leverages existing DNR calculation infrastructure from diagnostics module
  - Provides detailed split statistics including mean DNR, high-DNR counts, no-neighbor counts

- **MaxMinPicker Diversity Splitting** (`src/polyglotmol/models/api/core/splitting/diversity.py`)
  - Diversity-based splitting using RDKit's MaxMinPicker algorithm
  - Two operational modes:
    - **Friendly mode**: Diverse training set ensures broad chemical space coverage for learning
    - **Unfriendly mode**: Diverse test set creates most challenging generalization scenario
  - Multiple fingerprint support: Morgan (default), RDKit topological, MACCS keys
  - Configurable fingerprint parameters: radius, number of bits
  - Uses Tanimoto dissimilarity for maximum diversity selection
  - Reports similarity statistics: train/test avg similarity, cross-similarity
  - Integration with `DataSplitter` class via `strategy="maxmin"` parameter
  - Addresses diversity sampling requirements from molecular ML literature

- **Butina Clustering-Based Splitting** (`src/polyglotmol/models/api/core/splitting/butina.py`)
  - Leave-cluster-out cross-validation based on Tanimoto similarity clustering
  - Uses Butina's sphere exclusion algorithm (Butina 1999, J. Chem. Inf. Comput. Sci.)
  - Ensures chemically similar molecules stay together in either training or test set
  - Prevents information leakage from structural similarity, addressing MolAgent clustering strategy
  - Key features:
    - **Automatic clustering**: Self-adaptive cluster count based on similarity threshold
    - **Greedy balanced assignment**: Largest clusters assigned first to balance train/test sizes
    - **Leave-cluster-out validation**: Entire clusters move as units (no split within cluster)
  - Configurable parameters: similarity_threshold (default 0.6), fingerprint type, radius, nbits
  - Reports cluster statistics: n_clusters, cluster sizes, intra/inter-cluster similarity
  - Integration with `DataSplitter` class via `strategy="butina"` parameter
  - More fine-grained than scaffold splitting (considers full molecular topology vs core structure only)
  - Suitable for evaluating generalization to similar but unseen chemical combinations

- **Feature Clustering-Based Splitting** (`src/polyglotmol/models/api/core/splitting/feature_clustering.py`)
  - General-purpose clustering split supporting arbitrary molecular representations (not limited to fingerprints)
  - Three clustering algorithms:
    - **K-means++**: Fast spherical clustering with smart initialization
    - **Hierarchical**: Ward linkage tree-based clustering
    - **DBSCAN**: Density-based clustering with automatic cluster count detection
  - Flexible feature input sources:
    - **User-provided representations**: 3D embeddings (Boltz-2), language models (ChemBERTa), custom features
    - **RDKit descriptors**: ~20 physicochemical descriptors (MW, LogP, TPSA, etc.)
    - **Fingerprints**: Morgan/RDKit/MACCS as fallback
  - Automatic optimal k selection via Silhouette score maximization (k ∈ [2, √n])
  - Feature standardization with StandardScaler for fair distance computation
  - Comprehensive clustering quality metrics:
    - **Silhouette score**: [-1, 1], >0.5 indicates good clustering
    - **Calinski-Harabasz index**: Higher values indicate better-defined clusters
    - **Davies-Bouldin index**: Lower values indicate better separation
  - Leave-cluster-out assignment with greedy balanced strategy
  - DBSCAN noise point handling (assigned to training set)
  - Integration with `DataSplitter` class via `strategy="feature_clustering"` parameter
  - Configurable parameters: clustering_algorithm, n_clusters, auto_select_k, features, use_descriptors, dbscan_eps, dbscan_min_samples
  - Complementary to Butina splitting: Butina uses Tanimoto+fingerprints, feature_clustering uses Euclidean+arbitrary features
  - Ideal for non-fingerprint representations (3D structures, quantum features, embeddings)
  - Comprehensive test suite (14 passed, 2 DBSCAN skipped due to numpy compatibility)

- **Shared Splitting Utilities** (`src/polyglotmol/models/api/core/splitting/utils.py`)
  - Centralized utility functions to eliminate code duplication across splitting strategies
  - `compute_fingerprints()`: Unified fingerprint generation (Morgan, RDKit, MACCS)
  - `compute_avg_similarity()`: Average pairwise Tanimoto similarity within a set
  - `compute_cross_similarity()`: Average Tanimoto similarity between train and test sets
  - Reduces code duplication by 64 lines across butina.py and diversity.py
  - Shared by Butina, MaxMin, and Feature Clustering splitting strategies

- **Dataset Splitting Documentation** (`docs/source/usage/data/splitting.md`)
  - Comprehensive guide to all 10 supported splitting strategies (train_test, train_val_test, nested_cv, cv_only, scaffold, dnr, maxmin, butina, feature_clustering, user_provided)
  - Detailed implementation references with code locations and line numbers
  - Visual workflow diagrams for each splitting strategy
  - Scaffold split section with Bemis-Murcko vs Generic comparison and balanced vs random split methods
  - Feature Clustering section with complete examples:
    - User-provided features (3D embeddings, language models)
    - RDKit descriptors with K-means/Hierarchical clustering
    - DBSCAN with automatic cluster detection
    - Feature Clustering vs Butina comparison table
    - Typical workflow example with Boltz-2 embeddings
  - Drug discovery use cases and advantages over random splitting
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
- **Boltz-2 Test Failures and Registration** (`src/polyglotmol/representations/AI_fold/boltz2/`)
  - Fixed ligand_id truncation issue: Changed from 16-character cache_key to short ID 'L' to prevent Boltz-2 KeyError (`structure_predictor.py:157`)
  - Added AI_fold module import to `representations/__init__.py` to trigger Boltz2Embedder registration
  - Fixed test_registration: Corrected assertion to check instance instead of class (registry returns instances by design)
  - Fixed test_full_workflow: Corrected Dataset API usage (`feature_names`, `features.iloc[0]`) and `add_features()` argument order
  - Fixed VAE models indentation: Corrected MolecularVAE class `__init__` method indentation to prevent `RuntimeError: super(): no arguments`
  - Test results: 12 passed (up from 9), 1 skipped (GPU-dependent test validated manually with pre-computed CIF files)
  - All 4 originally failing tests now pass: `test_structure_prediction`, `test_caching_works`, `test_registration`, `test_full_workflow`

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

- **DNR and MaxMin Splitting Test Fixes** (`tests/models/splitting/test_dnr_maxmin_split.py`)
  - Fixed test dataset creation: Changed from `from_smiles_list()` to `from_df()` to properly set label_names
  - Fixed diversity.py: Changed `mol.rdkit_mol` to `mol.get_rdkit_mol()` method call
  - All 12 tests now pass successfully (previously 9 failed, 3 passed)
  - Verified DNR-based splitting with all 3 modes (threshold, quantile, neighbor)
  - Verified MaxMinPicker diversity splitting with friendly/unfriendly modes
  - Confirmed DataSplitter integration for both new strategies

### Changed
- **Dataset Quality Analysis Migration**: Replaced `overview.py` with comprehensive `diagnostics/` module
  - Migrated basic statistics and molecular weight calculations to `diagnostics/core.py`
  - Enhanced with DNR analysis and activity cliff detection capabilities
  - Improved visualization quality with publication-ready SVG output
  - Updated `src/polyglotmol/data/__init__.py` to export `DatasetDiagnostics` instead of `generate_dataset_report`

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
