"""Pytest configuration: skip integration test scripts.

Integration tests require MPI + patched libaims and are run via mpiexec
(see Makefile ``test-integration``), not via pytest.  This file prevents
pytest from importing them (which would trigger ``comm.Abort(1)`` at
module level if ``AIMSPY_TEST_AIMS_LIBPATH`` is unset).
"""

collect_ignore = [
    "test_baseline.py",
    "test_warmstart.py",
    "test_capture_overlap.py",
    "test_regression.py",
    "test_export_deeph.py",
    "test_strategies.py",
]
