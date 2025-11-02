PYTHON ?= python3

.PHONY: update-vscode-includes
update-vscode-includes:
	@echo "Updating VS Code C/C++ include paths from active Python environment..."
	$(PYTHON) .vscode/update_includes.py
	@echo "Done. Restart VS Code or reload window if needed."

.PHONY: dev
dev: install-dev

.PHONY: install-dev
install-dev:
	@echo "Installing package in editable mode and dev dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements-dev.txt
	@echo "Dev environment ready."

.PHONY: test
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest -q

.PHONY: lint
lint:
	@echo "Running ruff..."
	$(PYTHON) -m ruff check .

.PHONY: check
check: lint test
	@echo "All checks passed."
