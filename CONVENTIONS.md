# Development Conventions

This document outlines the development conventions to follow in the project to maintain code quality and facilitate collaboration.

• Follow Python best practices and adhere to PEP 8.
• Classes and functions must include docstrings (Google Style).
• Use type hints to specify the argument types and return values of functions.
• Use pyproject.toml to specify project dependencies and configurations.
• Use hatch for project management tasks.
• Use sphinx to generate documentation for the project.
• Use bump2version to manage the project version.
• Write and execute unit tests using pytest.
• Use ruff to automatically format the code.
• Keep the code as simple and readable as possible. If a function or class is hard to understand, it likely needs refactoring.
• Use descriptive names for variables, functions, and classes.
• The package must be compatible with Python 3.10 or higher.
• Use logging to log information, warnings, and errors in the code. Avoid using print.
