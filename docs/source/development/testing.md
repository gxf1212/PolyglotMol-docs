# Running Tests

PolyglotMol uses [pytest](https://docs.pytest.org/) for testing. Writing and running tests is crucial to ensure code correctness and prevent regressions.

## Prerequisites

* A working development environment (see :doc:`setup_dev_env`). Ensure all core and test dependencies are installed (`pip install -e .[tests]` or `pip install -e .[dev]`).

## Running Tests

1.  **Activate Environment:** Make sure your conda development environment (e.g., `polyglotmol-dev`) is activated.
    ```bash
    conda activate polyglotmol-dev
    ```

2.  **Navigate to Root Directory:** Change to the root directory of the PolyglotMol repository (the one containing `pyproject.toml` and the `tests` directory).

3.  **Run Pytest:** Execute the `pytest` command.

    ```bash
    # Run all tests
    pytest
    ```
    Pytest will automatically discover and run all files matching the pattern `test_*.py` or `*_test.py` inside the `tests` directory and its subdirectories.

    **Common Options:**
    * **Run tests in a specific file:**
      ```bash
      pytest tests/representations/fingerprints/test_rdkit.py
      pytest tests/representations/fingerprints/test_deepchem.py
      # Add other specific test files as needed
      ```
    * **Run specific test function:** `pytest tests/representations/fingerprints/test_rdkit.py::test_rdkit_fp_instantiation`
    * **Run tests with a specific marker:** `pytest -m "slow"` (if markers are used)
    * **Stop on first failure:** `pytest -x`
    * **Verbose output:** `pytest -v`
    * **Show print statements:** `pytest -s`
    * **Show local variables on failure:** `pytest -l`
    * **Run with coverage:** (Requires `pytest-cov`) `pytest --cov=polyglotmol --cov-report=html`

为了更好地管理需要网络连接或可选依赖的测试，我将使用 pytest.mark.skipif 来标记这些测试，而不是将文件拆分。这样你可以在运行时通过标记来选择性地运行测试（例如，运行 pytest -m "not internet" 来跳过需要网络的测试）。



## Writing Tests

* Tests should be placed in the `tests` directory, mirroring the structure of the `src/polyglotmol` directory where possible.
* Test filenames should start with `test_` or end with `_test.py`.
* Test function names should start with `test_`.
* Use clear and descriptive names for test functions.
* Use `assert` statements to check for expected outcomes.
* Use `pytest.raises` to check for expected exceptions.
* Use fixtures (`@pytest.fixture`) for setting up reusable test data or resources.
* Parameterize tests (`@pytest.mark.parametrize`) to run the same test logic with different inputs.
* Aim for good test coverage for any new code added.

Refer to the [pytest documentation](https://docs.pytest.org/) for more details on writing tests.
