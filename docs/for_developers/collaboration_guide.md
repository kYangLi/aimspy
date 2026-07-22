# Collaboration Guide

Welcome to the AimsPy community! We believe that groundbreaking tools are built together. This guide outlines how you can contribute, whether through code, documentation, ideas, or community support.

## Ways to Contribute

There are many ways to contribute, all of which are highly valued:

* **Code & Features**: Implement new functionalities, fix bugs, or improve performance. Particularly welcome are new `ExternalMatrixSource` adapters for DFT codes beyond FHI-aims, and new `Strategy` variants.
* **Documentation & Examples**: Improve guides, fix typos, or add tutorials and use cases.
* **Community Help**: Answer questions on discussions, report bugs, or suggest features.
* **Spread the Word**: Star our [GitHub repository](https://github.com/kYangLi/aimspy) to show your support.

For major new features, please open a **GitHub Discussion** or **Issue** first to align with the project's direction and architecture.

## Development Workflow Overview

To maintain a clean and modular codebase, please follow this high-level workflow when contributing code:

1. **Fork & Clone**: Fork the official repository and clone your fork locally.
2. **Architectural Alignment**: Understand which layer your feature belongs to (public API / interface adapter / callback framework / ctypes binding / FHI-aims patch). For technical details, please see the [Development Guide](./development_guide.md).
3. **Implement & Test**: Develop your feature in a dedicated branch, following our coding standards and including tests.
4. **Submit Pull Request**: Push your branch and open a PR against the main repository's `main` branch.

## Fork and Pull Request Process

1. **Fork the Repository**: Click the 'Fork' button on the [AimsPy GitHub page](https://github.com/kYangLi/aimspy).
2. **Create a Feature Branch**:

    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **Commit Your Changes**: Use clear, descriptive commit messages.

    ```bash
    git commit -m "feat(interface): add OpenMX adapter for ExternalMatrixSource"
    ```

4. **Push to Your Fork**:

    ```bash
    git push origin feature/YourFeatureName
    ```

5. **Open a Pull Request**: Navigate to your fork on GitHub and click "Compare & pull request". Fill in the PR template describing your changes.

**Quality Checklist for PRs:**

* Code follows the existing style (`ruff check .` and `black --check .` pass; `make lint` runs both).
* New features include relevant tests. Unit tests go in `tests/unit/`; integration tests go in `tests/` and are gated behind `AIMSPY_TEST_AIMS_LIBPATH`.
* All tests pass (`make test` for unit, `make test-integration` for integration).
* Public API changes are reflected in `docs/` and the README.
* If you touched the FHI-aims patch, the patch version is bumped and `aimspy patch --check` still applies cleanly on a fresh checkout.

## Contributor License Agreement (CLA)

To accept your contributions, we require a Contributor License Agreement. This agreement confirms that you have the right to grant us permission to use your contribution. It protects you, the maintainers, and all users of the project.

## Getting Help

If you have questions during the process:

* Check the [Development Guide](./development_guide.md) for technical concepts.
* Start a **GitHub Discussion**.
* Review existing **Issues** and **Pull Requests**.

Thank you for your interest in making AimsPy better!
