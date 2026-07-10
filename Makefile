.PHONY: clean install test tests build lint help patch generate-patch

VENV := .venv
.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install          Create .venv, install in editable mode with dev deps"
	@echo "  test             Run tests"
	@echo "  build            Build distribution wheel"
	@echo "  lint             Run ruff check + black"
	@echo "  patch            Apply the bundled FHI-aims patch to AIMS_SOURCE"
	@echo "  generate-patch   Regenerate aimspy/patches/fhiaims_250822_1.patch from binding/"
	@echo "  clean            Remove build artifacts and cache files"
	@echo "  help             Show this help message"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev dependencies..."
	uv pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest tests -q

lint:
	.venv/bin/ruff check .
	.venv/bin/black --check .

build:
	uv build --wheel -o dist ./

patch:
	.venv/bin/aimspy patch "$${AIMS_SOURCE:-./FHIaims250822_1}"

clean:
	@echo "Cleaning build artifacts and cache files..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./FHIaims*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./FHIaims*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.mod" -not -path "./.venv/*" -not -path "./FHIaims*" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage dist build binding/build aimspy_build 2>/dev/null || true
	rm -f aimspy/_aims*.so aimspy/_aims*.pyd 2>/dev/null || true
	@echo "Clean complete!"
