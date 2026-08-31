.PHONY: clean install test test-baseline test-warmstart test-capture-overlap \
        test-regression test-export-deeph test-strategies test-integration \
        test-all test-memory-loop test-dHde-capture test-dHde-inject-direct \
        test-dHde-inject-defer test-dHde run-from-scratch run-continue-calc run-example \
        test-grid-data-capture test-basis-export test-basis-callback-paths \
        test-observables \
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
	@echo "  test-dHde-capture    Run DFPT dH/de capture + export (produces deeph_dHde_out/)"
	@echo "  test-dHde-inject-direct  Run DFPT dH/de direct warmstart (needs deeph_dHde_out/)"
	@echo "  test-dHde-inject-defer   Run DFPT dH/de deferred warmstart (needs deeph_dHde_out/)"
	@echo "  test-dHde             Run all 3 dHde tests in order"
	@echo "  test-dHde-serial-capture  Run DFPT serial dH/de capture (produces deeph_dHde_serial_out/)"
	@echo "  test-dHde-serial-inject   Run DFPT serial dH/de warmstart (needs deeph_dHde_serial_out/)"
	@echo "  test-grid-data-capture Run real-space grid capture test (MoS2_LDA)"
	@echo "  test-basis-export     Run NAO basis export test (MoS2_LDA)"
	@echo "  test-basis-callback-paths  Run basis callback registration-semantics test (MoS2_LDA)"
	@echo "  test-observables     Run final energy/force/analytical-stress API test"
	@echo "  test-integration     Run all integration tests in dependency order"
	@echo "  test-memory-loop     Run 15-cycle memory pressure test (capture_overlap=True)"
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
	@echo "  AIMSPY_MEM_LOOP_N         Memory loop iterations (default: 15)"
	@echo "  AIMSPY_MEM_LOOP_THRESHOLD_KB  Max RSS drift in last 5 iters (default: 100MB)"
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

test-memory-loop:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_memory_loop.py

test-dHde-capture:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_dHde_capture.py

test-dHde-inject-direct:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_dHde_inject_direct.py

test-dHde-inject-defer:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_dHde_inject_defer.py

test-dHde: test-dHde-capture test-dHde-inject-direct test-dHde-inject-defer

test-dHde-serial-capture:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_dHde_serial_capture.py

test-dHde-serial-inject:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_dHde_serial_inject.py

test-dHde-serial: test-dHde-serial-capture test-dHde-serial-inject

test-callback-reset:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_callback_reset.py

test-grid-data-capture:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_grid_data_capture.py

test-basis-export:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_basis_export.py

test-basis-callback-paths:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_basis_callback_paths.py

test-observables:
	ulimit -s unlimited && mpiexec -np $${AIMSPY_TEST_NPROC:-8} python tests/test_observables.py

test-integration: test-baseline test-export-deeph test-warmstart \
                  test-capture-overlap test-regression test-strategies test-dHde \
                  test-dHde-serial test-callback-reset test-grid-data-capture \
                  test-basis-export test-basis-callback-paths test-observables

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
	rm -rf tests/data/MoS2_DFPT/deeph_dHde_out 2>/dev/null || true
	rm -f tests/data/MoS2/*.out 2>/dev/null || true
	rm -f tests/data/MoS2_DFPT/*.out 2>/dev/null || true
	rm -f tests/data/MoS2_DFPT/*.dat 2>/dev/null || true
	rm -f tests/data/MoS2_DFPT/*.npy 2>/dev/null || true
	rm -rf examples/*/deeph_data 2>/dev/null || true
	rm -f examples/*/*.out 2>/dev/null || true
	@echo "Clean complete!"
