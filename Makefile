install:
	uv pip install -r requirements.txt

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
