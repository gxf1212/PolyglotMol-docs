# Setting Up Development Environment

To contribute code or run tests locally, you need to set up a development environment.

## Prerequisites

* **Git:** For version control.
* **Python:** Version >= 3.9 (as specified in `pyproject.toml`).
* **Conda (Recommended):** For managing environments and complex dependencies like RDKit.
* **Java Development Kit (JDK):** Version >= 11, required for `CDK-pywrapper`. Ensure `JAVA_HOME` is set correctly or Java is in your system PATH.

## Steps

1.  **Clone the Repository:**
    If you haven't already, clone your fork of the repository (see :doc:`contributing` guide) or the main repository:
    ```bash
    # If contributing via fork:
    git clone [https://github.com/gxf1212/PolyglotMol.git](https://github.com/gxf1212/PolyglotMol.git) # Using your username

    # Or clone the main repo directly (replace OWNER if different):
    # git clone [https://github.com/gxf1212/PolyglotMol.git](https://github.com/gxf1212/PolyglotMol.git)
    cd PolyglotMol
    ```

2.  **Create and Activate Conda Environment (Recommended):**
    It's highly recommended to use a dedicated conda environment to manage dependencies, especially RDKit.

    ```bash
    # Create a new environment named 'polyglotmol-dev' with Python 3.9
    conda create -n polyglotmol-dev python=3.9 -y

    # Activate the environment
    conda activate polyglotmol-dev

    # Install RDKit (using conda-forge channel is usually recommended)
    conda install -c conda-forge rdkit -y

    # (Optional but recommended) Install DeepChem via conda if available/preferred
    # conda install -c conda-forge deepchem -y
    # Or install via pip later
    ```
    *Note: Adjust the Python version if needed, but ensure it matches `requires-python`.*

3.  **Install Package in Editable Mode with Dev Dependencies:**
    Navigate to the root directory of the cloned repository (where `pyproject.toml` is located) and run:

    ```bash
    # Ensure your conda environment is active
    pip install -e .[dev]
    ```
    * `-e .`: Installs the package in "editable" mode, meaning changes you make to the source code in the `src/` directory will be immediately reflected when you import the package, without needing to reinstall.
    * `[dev]`: Installs the dependencies listed under `[project.optional-dependencies.dev]` in `pyproject.toml` (which should include `pytest`, linters, formatters, etc.). If you installed DeepChem via conda, pip might skip it or handle version conflicts.

4.  **(Optional) Set up Pre-commit Hooks:**
    If the project uses pre-commit for code formatting and linting checks:
    ```bash
    # Install pre-commit (if not included in [dev] dependencies)
    # pip install pre-commit

    # Set up the git hooks
    pre-commit install
    ```
    This will automatically run checks (like Black, Ruff) before you commit changes.

Your development environment is now ready! You can edit the code in `src/polyglotmol`, run tests, and build the documentation locally.
