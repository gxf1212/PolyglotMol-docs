# Contributing to PolyglotMol

We welcome contributions to PolyglotMol! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code changes, your help is appreciated.

## Ways to Contribute

* **Reporting Bugs:** If you find a bug, please open an issue on our GitHub repository. Include details about the error, how to reproduce it, your environment (OS, Python version, package versions), and any relevant code snippets or tracebacks.
* **Suggesting Enhancements:** Have an idea for a new feature or an improvement to an existing one? Open an issue to discuss it.
* **Improving Documentation:** Found a typo, unclear explanation, or missing information in the docs? Feel free to open an issue or submit a pull request with corrections.
* **Adding Code:** If you want to add new featurizers, fix bugs, or implement new features, please follow the development workflow below.

## Development Workflow

1.  **Fork the Repository:** Create your own fork of the main PolyglotMol repository on GitHub (github.com/gxf1212/PolyglotMol).
2.  **Clone Your Fork:** Clone your forked repository to your local machine:
    ```bash
    git clone [https://github.com/gxf1212/PolyglotMol.git](https://github.com/gxf1212/PolyglotMol.git) # Using your username
    cd PolyglotMol
    ```
3.  **Set up Development Environment:** Follow the instructions in the :doc:`setup_dev_env` guide.
4.  **Create a Branch:** Create a new branch for your changes:
    ```bash
    git checkout -b my-feature-branch # Or fix/bug-name, etc.
    ```
5.  **Make Changes:** Implement your feature, fix the bug, or improve the documentation.
6.  **Write Tests:** Add tests for any new code functionality. Ensure existing tests pass. See the :doc:`testing` guide.
7.  **Format and Lint:** Ensure your code conforms to the project's style guidelines (e.g., using Black, Ruff, or pre-commit hooks if configured).
8.  **Update Documentation:** If you added or changed features, update the relevant documentation pages (both usage guides and API references/docstrings).
9.  **Commit Changes:** Commit your changes with clear and concise commit messages.
10. **Push to Your Fork:**
    ```bash
    git push origin my-feature-branch
    ```
11. **Submit a Pull Request:** Go to the main PolyglotMol repository on GitHub (github.com/gxf1212/PolyglotMol) and open a pull request from your branch to the main development branch (e.g., `main` or `develop`). Provide a clear description of your changes.

## Code Style

(To be defined - e.g., We follow PEP 8 and use Black for formatting.)

## Questions?

Feel free to open an issue if you have questions about contributing.
