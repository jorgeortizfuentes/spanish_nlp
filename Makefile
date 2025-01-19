install:
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt

venv:
	@echo "Setting up virtual environment..."
	@rm -rf .venv
	uv venv .venv
	uv pip install -U pip
	@echo "Run 'source .venv/bin/activate' to activate the virtual environment."

install_local: 
	@echo "Installing package in editable mode..."
	uv pip install -e .
	@echo "Package installed successfully. You can now use the package."

.PHONY: aider
aider:
	@echo "Setting up environment variables..."
	@echo "VERTEXAI_PROJECT=$(VERTEXAI_PROJECT)"
	@echo "VERTEXAI_LOCATION=$(VERTEXAI_LOCATION)"
	@echo "Executing 'aider'..."
	aider

test:
	@echo "Creating outputs directory..."
	@mkdir -p outputs
	@echo "Running tests with coverage..."
	pytest --cov=spanish_nlp --cov-report=html:outputs/coverage --cov-report=term-missing -v tests/ | tee outputs/pytest-report.txt
