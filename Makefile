.PHONY: clean install test test-baseline test-warmstart test-capture-overlap \
        test-regression test-export-deeph test-strategies test-integration \
        test-all run-from-scratch run-continue-calc run-example \
        build lint help patch

VENV := .venv
.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install              Create .venv, install in editable mode with dev deps"
	@echo "  test                 Run unit tests (pytest -v, no MPI required)"
	@echo "  test-baseline        Run baseline SCF (produces rs_hamiltonian.out etc.)"
	@echo "  test-export-deeph    Run SCF + DeepH export (produces deeph_out/)"
	@echo "  test-warmstart       Run warmstart test (needs rs_hamiltonian.out + deeph_out/)"
	@echo "  test-capture-overlap Run overlap capture test (no prerequisites)"
	@echo "  test-regression      Run regression test (needs rs_hamiltonian.out + deeph_out/)"
	@echo "  test-strategies      Run strategy test (needs rs_hamiltonian.out + deeph_out/)"
	@echo "  test-integration     Run all 6 integration tests in dependency order"
	@echo "  test-all             Run unit + integration tests"
	@echo "  run-from-scratch     Run H2O baseline SCF + DeepH export example"
	@echo "  run-continue-calc    Run H2O warmstart example (needs run-from-scratch first)"
	@echo "  run-example          Run all examples in order"
	@echo "  build                Build sdist + wheel"
	@echo "  lint                 ruff check + black"
	@echo "  patch                Apply the bundled FHI-aims patch to AIMS_SOURCE"
	@echo "  clean                Remove build artifacts and cache files"
	@echo "  help                 Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  AIMSPY_TEST_AIMS_LIBPATH  Path to patched libaims.so (required for integration)"
	@echo "  AIMSPY_TEST_NPROC         MPI process count (default: 8)"
	@echo ""
	@echo "Prerequisites for integration tests:"
	@echo "  source /path/to/intel/setvars.sh   (Intel OneAPI for MPI + MKL)"
	@echo "  ulimit -s unlimited                (handled automatically by each target)"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev dependencies..."
	uv pip install -e ".[dev]"

test:
	python -m pytest -v

test-baseline:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_baseline.py

test-export-deeph:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_export_deeph.py

test-warmstart:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_warmstart.py

test-capture-overlap:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_capture_overlap.py

test-regression:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_regression.py

test-strategies:
	ulimit -s unlimited && python tests/test_strategies.py

test-integration: test-baseline test-export-deeph test-warmstart \
                  test-capture-overlap test-regression test-strategies

test-all: test test-integration

run-from-scratch:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python examples/from_scratch/run.py

run-continue-calc:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python examples/continue_calc/run.py

run-example: run-from-scratch run-continue-calc

lint:
	ruff check .
	black --check .

build:
	uv build --sdist --wheel -o dist ./

patch:
	aimspy patch "$${AIMS_SOURCE:-./FHIaims250822_1}"

clean:
	@echo "Cleaning build artifacts and cache files..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./FHIaims*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./FHIaims*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.mod" -not -path "./.venv/*" -not -path "./FHIaims*" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage dist build 2>/dev/null || true
	rm -f aimspy/_aims*.so aimspy/_aims*.pyd 2>/dev/null || true
	rm -rf tests/data/MoS2/deeph_out tests/data/MoS2/_regression_* 2>/dev/null || true
	rm -f tests/data/MoS2/*.out 2>/dev/null || true
	rm -rf examples/*/deeph_data 2>/dev/null || true
	rm -f examples/*/*.out 2>/dev/null || true
	@echo "Clean complete!"
