# Developer Guide

This document describes the steps for updating the package version and the changelog, and how the publishing process works.

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and virtual environments. Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`, which must be committed.

```bash
# Create .venv and install runtime + dev dependencies from uv.lock
uv sync
# Also install docs and notebook dependencies
uv sync --all-groups
# Run the test suite
make test
```

To add or remove dependencies, use `uv add <package>` (runtime), `uv add --group dev <package>` (development) or `uv remove <package>`. These commands update both `pyproject.toml` and `uv.lock`; never edit `uv.lock` by hand.

## Updating the Package Version

This project uses `hatch` for version management (run through `uvx`, so it does not need to be installed). The version is stored in `src/spanish_nlp/__about__.py`.

To update the version, use the `uvx hatch version` command. You can specify the new version directly or use semantic increments (patch, minor, major).

**Options:**

1.  **Specify the exact version:**

    ```bash
    uvx hatch version <new_version>
    # Example:
    uvx hatch version 0.4.0
    ```

2.  **Increment semantically:**
    - Increment patch: `0.3.1` -> `0.3.2`
      ```bash
      uvx hatch version patch
      ```
    - Increment minor version: `0.3.1` -> `0.4.0`
      ```bash
      uvx hatch version minor
      ```
    - Increment major version: `0.3.1` -> `1.0.0`
      ```bash
      uvx hatch version major
      ```

While the package is in `0.x`, breaking changes (dropping a Python version, raising dependency minimums, changing public behavior) bump the **minor** version; fixes and internal changes bump the **patch** version.

The version bump and the changelog update are done together, in a single commit on `develop`, right before opening the release PR to `main` (see [Preparing a Release](#preparing-a-release)). Do not bump the version in feature branches.

## Updating the Changelog

`CHANGELOG.md` is generated from the Git history with [git-cliff](https://git-cliff.org), configured in `cliff.toml`. Never edit the generated sections by hand; fix the commit messages instead.

git-cliff relies on [Conventional Commits](https://www.conventionalcommits.org). Each commit type is placed in a changelog section:

| Commit prefix         | Changelog section       |
| --------------------- | ----------------------- |
| `feat`                | 🚀 Features             |
| `fix`                 | 🐛 Bug Fixes            |
| `refactor`            | 🚜 Refactor             |
| `doc`/`docs`          | 📚 Documentation        |
| `perf`                | ⚡ Performance          |
| `style`               | 🎨 Styling              |
| `test`                | 🧪 Testing              |
| `chore`, `ci`         | ⚙️ Miscellaneous Tasks  |
| `revert`              | ◀️ Revert               |
| anything else (`build`, ...) | 💼 Other     |

Commits starting with `chore(release): prepare for`, `chore(changelog):` or `chore(deps...)` are left out of the changelog.

git-cliff runs through `uvx`, so it does not need to be installed:

```bash
# Preview the unreleased changes without writing any file
uvx git-cliff --unreleased
# Regenerate CHANGELOG.md, labelling unreleased commits as the given version
uvx git-cliff --tag v0.5.0 -o CHANGELOG.md
```

Always pass `--tag` when preparing a release. Without it, the new commits are listed under `[unreleased]` because the `vX.Y.Z` tag does not exist yet.

## Preparing a Release

1.  Make sure every feature branch for the release is merged into `develop`, and update your local copy:
    ```bash
    git checkout develop
    git pull origin develop
    ```
2.  Bump the version and regenerate the changelog with the same version:
    ```bash
    uvx hatch version 0.5.0
    uvx git-cliff --tag v$(uvx hatch version) -o CHANGELOG.md
    ```
3.  Review `src/spanish_nlp/__about__.py` and `CHANGELOG.md`. If the release contains breaking changes, make sure they are easy to spot in the release notes.
4.  Commit both files together and push:
    ```bash
    git add src/spanish_nlp/__about__.py CHANGELOG.md
    git commit -m "chore(release): prepare for v$(uvx hatch version)"
    git push origin develop
    ```
5.  Open a Pull Request from `develop` to `main` (see [Publishing to PyPI](#publishing-to-pypi)).

## Contribution Workflow (Gitflow)

This project follows the Gitflow workflow for managing branches and contributions.

1.  **Main Branch (`main`):** Represents the latest stable release. Direct commits to `main` are **prohibited**. Releases are tagged from this branch after merging from `develop`.
2.  **Development Branch (`develop`):** This is the primary integration branch for ongoing development. All feature branches must be merged into `develop` first.
3.  **Feature Branches (`feature/<feature-name>`):** Create these branches **from `develop`** for new features or significant changes. Use the naming convention `feature/nombre-descriptivo-de-la-feature`.
4.  **Pull Requests (PRs):**
    - **Feature to Develop:** When a feature is complete, create a Pull Request (PR) from your `feature/<feature-name>` branch back to the `develop` branch.
    - **Develop to Main:** For releases, create a Pull Request (PR) from the `develop` branch to the `main` branch. This merge triggers the automated publishing process.
    - Ensure your code adheres to project conventions (see [Development Conventions](CONVENTIONS.md)) and passes all tests (`make test`).
    - All PRs require review before merging.

## Publishing to PyPI

Publishing to PyPI is **automated** using GitHub Actions (`.github/workflows/main.yml`).

**The process is as follows:**

1.  When changes are merged (usually via Pull Request) into the `main` branch.
2.  The GitHub Actions workflow is automatically triggered.
3.  Tests are run (`make test`).
4.  If tests pass, the package is built (`uv build`).
5.  The version is extracted from the built package.
6.  The package is published to PyPI using the `secrets.PYPI_API_TOKEN`.
7.  If publishing is successful, a Git tag is automatically created in the repository with the format `vX.Y.Z` (e.g., `v0.4.0`).

**Therefore, to publish a new version:**

1.  Ensure the `develop` branch contains all the features and fixes for the release.
2.  Bump the version and regenerate the changelog in `develop` (see [Preparing a Release](#preparing-a-release)).
3.  Commit and push the release commit to `develop`.
4.  Create a Pull Request from `develop` to `main`.
5.  Once the PR is reviewed and approved, **merge it into `main`**. This merge will trigger the automated publishing workflow.

**You do not need to run `uv publish` manually.**
