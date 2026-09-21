## [0.4.0] - 2025-04-26

### 🚀 Features

- Add spell checker with dictionary and contextual LM backends
- Refactor contextual checker for hybrid LM/dictionary correction
- Add skeleton for ContextualLMSpellChecker

### 🐛 Bug Fixes

- Update dependencies to resolve spacy/pydantic error
- Handle None returned by pyspellchecker candidates
- Remove optimization skipping LM for dict-correct words
- Adjust spellchecker tests to match pyspellchecker behavior

### 💼 Other

- Update dependencies versions in requirements.txt
- Update spellchecking notebook outputs text examples markdown and execution counts
- Remove contextual language model example from spellchecking notebook
- Update readme
- Remove references to sphinx and bump2version from development conventions
- Bump version to 0.4.0
- Update development dependencies in requirements-dev.txt

### 🚜 Refactor

- Remove non-functional contextual_lm spellchecker

### 📚 Documentation

- Execute Spellchecking notebook and save outputs
- Separate examples for custom distance and dictionary in notebook
- Update custom dictionary example in Spellchecking notebook
- Update spellchecking example notebook
- Add spell checker documentation to README
- Update table of contents
- Add developer guide
- Document Gitflow contribution workflow in README-dev
- Clarify Gitflow process and main branch protection
- Clarify Gitflow workflow and PR process in CONTRIBUTING
- Add reference to CONTRIBUTING.md in README
- Add reference to CONVENTIONS.md in CONTRIBUTING.md
- Improve formatting and update instructions in contributing guide
- Add PyPI downloads badge to README
- Add maintenance and license badges to README

### 🧪 Testing

- Add tests for dictionary spell checker
- Fix spellchecker tests to match dictionary behavior

### ⚙️ Miscellaneous Tasks

- Rename README-dev.md to CONTRIBUTING.md
- Trigger workflow only on main branch push
- Run tests on develop branch
- Run tests on feature branches, build only on main
- Remove comments from main.yml
## [0.3.1] - 2025-01-20

### 💼 Other

- Requires python version

### 📚 Documentation

- Enhance README for better presentation and clarity
- Update README structure and contact details
- Update README formatting and installation info

### ⚙️ Miscellaneous Tasks

- Bump version to 0.3.1
## [0.3.0] - 2025-01-19

### 🚀 Features

- Conventions and Makefile
- Add ipywidgets to dev requirements
- Add pytest target to Makefile
- Save test outputs to outputs dir and add coverage
- Add uv installation to GitHub Actions workflow

### 🐛 Bug Fixes

- Update init to include preprocess module
- Import SpanishPreprocess in test file
- Activate virtual environment before build
- Adjust CI workflow for editable install and build

### 💼 Other

- Requirements
- Requirements-dev to makefile
- Requirements-dev to makefile and set python version
- Numpy 1.26.4
- Move classifiers to a folder
- Refactor classifiers code
- Swifter
- Outputs to gitignore
- Run new examples
- Version to 0.3.0
- Create venv in workflow

### 🚜 Refactor

- Preprocess
- Use logging instead of print in transform method
- Use Makefile for running tests
- Simplify dependency installation in CI workflow

### ⚙️ Miscellaneous Tasks

- Use Makefile for dependency installation in CI
- Run workflow on develop and feature branches, publish on main
## [0.2.1] - 2023-02-26
