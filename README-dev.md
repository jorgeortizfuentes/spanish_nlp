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
    git push origin main
    ```

## Publishing to PyPI

Publishing to PyPI is **automated** using GitHub Actions (`.github/workflows/main.yml`).

**The process is as follows:**

1.  When changes are pushed to the `main` branch.
2.  The GitHub Actions workflow is automatically triggered.
3.  Tests are run (`make test`).
4.  If tests pass, the package is built (`hatchling build`).
5.  The version is extracted from the built package.
6.  The package is published to PyPI using the `secrets.PYPI_API_TOKEN`.
7.  If publishing is successful, a Git tag is automatically created in the repository with the format `vX.Y.Z` (e.g., `v0.4.0`).

**Therefore, to publish a new version, you only need to:**

1.  Update the version using `hatch version` (as described above).
2.  Commit and push the changes to the `main` branch.

**You do not need to run `hatch publish` manually.**
