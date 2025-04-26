# Developer Guide

This document describes the steps for updating the package version and how the publishing process works.

## Updating the Package Version

This project uses `hatch` for version management. The version is stored in `src/spanish_nlp/__about__.py`.

To update the version, use the `hatch version` command. You can specify the new version directly or use semantic increments (patch, minor, major).

**Options:**

1.  **Specify the exact version:**
    ```bash
    hatch version <new_version>
    # Example:
    hatch version 0.4.0
    ```

2.  **Increment semantically:**
    *   Increment patch: `0.3.1` -> `0.3.2`
        ```bash
        hatch version patch
        ```
    *   Increment minor version: `0.3.1` -> `0.4.0`
        ```bash
        hatch version minor
        ```
    *   Increment major version: `0.3.1` -> `1.0.0`
        ```bash
        hatch version major
        ```

**Steps:**

1.  Ensure you are on the main branch (`main`) and have the latest changes:
    ```bash
    git checkout main
    git pull origin main
    ```
2.  Run the `hatch version` command with the desired option:
    ```bash
    # Example for a new minor version
    hatch version minor
    ```
3.  Verify that the `src/spanish_nlp/__about__.py` file has been updated correctly.
4.  Add the change to Git staging, commit, and push the changes:
    ```bash
    git add src/spanish_nlp/__about__.py
    # Use the updated version in the commit message
    git commit -m "Bump version to $(hatch version)"
    # Push the change to the development branch
    git push origin develop
    ```
    *Note: After pushing to `develop`, you might need to create a Pull Request to merge `develop` into `main` to trigger the release.*

## Contribution Workflow (Gitflow)

This project follows the Gitflow workflow for managing branches and contributions.

1.  **Main Branch (`main`):** Represents the latest stable release. Direct commits to `main` are **prohibited**. Releases are tagged from this branch after merging from `develop`.
2.  **Development Branch (`develop`):** This is the primary integration branch for ongoing development. All feature branches must be merged into `develop` first.
3.  **Feature Branches (`feature/<feature-name>`):** Create these branches **from `develop`** for new features or significant changes. Use the naming convention `feature/nombre-descriptivo-de-la-feature`.
4.  **Pull Requests (PRs):**
    *   **Feature to Develop:** When a feature is complete, create a Pull Request (PR) from your `feature/<feature-name>` branch back to the `develop` branch.
    *   **Develop to Main:** For releases, create a Pull Request (PR) from the `develop` branch to the `main` branch. This merge triggers the automated publishing process.
    *   Ensure your code adheres to project conventions (see [Development Conventions](CONVENTIONS.md)) and passes all tests (`make test`).
    *   All PRs require review before merging.

## Publishing to PyPI

Publishing to PyPI is **automated** using GitHub Actions (`.github/workflows/main.yml`).

**The process is as follows:**

1.  When changes are merged (usually via Pull Request) into the `main` branch.
2.  The GitHub Actions workflow is automatically triggered.
3.  Tests are run (`make test`).
4.  If tests pass, the package is built (`hatchling build`).
5.  The version is extracted from the built package.
6.  The package is published to PyPI using the `secrets.PYPI_API_TOKEN`.
7.  If publishing is successful, a Git tag is automatically created in the repository with the format `vX.Y.Z` (e.g., `v0.4.0`).

**Therefore, to publish a new version:**

1.  Ensure the `develop` branch contains all the features and fixes for the release.
2.  Update the version in the `develop` branch using `hatch version` (as described above).
3.  Commit and push the version bump to `develop`.
4.  Create a Pull Request from `develop` to `main`.
5.  Once the PR is reviewed and approved, **merge it into `main`**. This merge will trigger the automated publishing workflow.

**You do not need to run `hatch publish` manually.**
