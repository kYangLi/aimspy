# Changelog

All notable changes to **aimspy** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-19

### Added

- Unified `Calculator.modify_init_ham()` API supporting both direct source
  and deferred decorator modes (replaces the former
  `ModifyInitialHamiltonianConfig` + `CalculatorConfig.modify` pair).
- `CalcState.FAILED` and `Calculator.force_close()` for safe recovery after
  SCF or `init()` failure (swallows Fortran errors, clears all state).
- `CallbackName` enum and `ExternalMatrixSource` `Protocol` for type-safe
  callback registration and pluggable matrix sources.
- `CalculatorConfig.capture_overlap=True` flag — live overlap matrix on all
  ranks via the `export_ovlp` callback (no longer rank-0-only fallback).
- `CalculatorConfig.initializer` hook — `fn(Calculator) -> None` invoked on
  rank 0 before `aimspy_init`.
- `AimspyCallbackError.callback_errors` attribute — preserves
  `(name, exception, traceback_str)` tuples for post-mortem inspection.
- Unit test suite in `tests/unit/` (57 tests, no MPI/libaims required):
  `test_structure`, `test_protocol_enum`, `test_poscar`, `test_deeph_data`.
- `tests/test_strategies.py` — `Strategy.ADD` / `SCALE` / `CUSTOM` via
  sub-MPI dispatch (FHI-aims is a global Fortran singleton).
- `tests/test_capture_overlap.py` — live overlap on all ranks + two-step API.
- `AIMSPY_TEST_AIMS_LIBPATH` environment variable for tests and examples
  (replaces hardcoded local `libaims.so` paths).
- `aimspy patch` CLI: `--check` / `--dry-run`, `--list`, `--no-git`, `-y`
  options and versioned bundled diffs.
- `pyproject.toml`: `Changelog` project URL.

### Changed

- Forces are now captured **before** the callback error check, so they are
  no longer lost when a callback raises.
- `CalcState` transitions: `init()` / `calc()` now wrap Fortran calls in
  try/except to transition to `FAILED` on errors.
- `Calculator.__exit__` uses `force_close()` on exception body to avoid
  masking the original error.
- Structure derived properties (`phase_factor`, `basis_subidx`,
  `orbit_per_atom`, `atom_permutation`) cached via
  `@functools.cached_property`.
- `basis_subidx` / `orbit_per_atom` vectorized (no Python loops).
- State guards added to `info`, `structure`, `forces`, `overlap` properties.
- Logging: INFO/WARNING emitted on rank 0 only; ERROR on all ranks.
- `register_callback` from `DONE` state now emits a `UserWarning` (the
  callback will not fire).
- Deferred `modify_init_ham` source: explicit `None` check raises
  `AimspyConfigError`.
- All `mpirun` references in docs and examples replaced with `mpiexec`.
- `Makefile build` target now produces both sdist and wheel (matches the
  `publish.yaml` workflow).
- `pyproject.toml` `Development Status` remains `3 - Alpha`.

### Fixed

- `AimspyInfo.frac_coords` units bug — was multiplied by `BOHR_TO_ANG`,
  now dimensionless.
- `np.maximum` merge in `_aimspy_blocks_to_poscar` silently dropped
  duplicate keys — now raises on duplicate.
- `DeepHData._build_elements_orbital_map`: per-shell `l` is no longer
  duplicated for multi-atom elements.
- `DeepHData` methods now raise `AimspyConfigError` (instead of
  `ValueError`) for consistency with the rest of the package.
- Removed dead code (`_map_to_center_cell`) and stale E741 lint warnings.

## [0.1.0] - 2026-07-10

### Added

- Full `aimspy` package (22 Python files, ~2950 LOC):
  - `calculator.py`: `Calculator` with `modify_h0()`, `capture_h0`, state
    machine.
  - `structure.py`: `AimspyStructure` (shared structure + orbital
    descriptor).
  - `matrix.py`: `AimspyMatrix` + aims↔aimspy CSR conversion.
  - `data.py`: `AimspyInfo`, `CsrMatrixDescriptor`.
  - `_callbacks/`: `CallbackSpec` / `CallbackManager` + 5 registered
    callbacks (`get_descr`, `export_ovlp`, `export_h0`, `modify_h0`,
    `python_func`).
  - `_binding/`: ctypes prototypes, Fortran structure mirrors, `CFUNCTYPE`
    types, `libloader` (with MPICH symbol-visibility workaround).
  - `interface/`: `ExternalMatrixSource` ABC.
  - `interface/deeph/`: `DeepHData`, `DeepHSource`, deeph↔aimspy
    converters.
- Bundled FHI-aims patch (`aimspy-patch_v0.1.0.diff`, 1105 lines) adding
  `src/aimspy_api/` (5 Fortran modules) and injection points in
  `initialize_scf.f90` / `scf_solver.f90` / `pbc_lists.f90`.
- `aimspy patch` CLI for applying / uninstalling / listing versioned
  patches (Click-based, with `git apply` and `patch -p1` backends).
- `AimspyInfo` ctypes mirror of the Fortran `TAimspyInfo` struct with
  automatic unit conversions (Bohr→Å, 1-based→0-based indices).
- Integration tests on MoS₂: `test_baseline`, `test_warmstart`,
  `test_regression` (50 checks), `test_export_deeph`.
- PyPI trusted-publishing workflow
  (`.github/workflows/publish.yaml`, triggered on GitHub release).
- `examples/from_scratch/run.py` — H₂O baseline SCF + DeepH export.

## [0.0.2] - 2026-06-18

### Changed

- Require Python 3.12–3.14.
- README: add PyPI badges, fix DeepX/DeepH-pack link and name.
- `pyproject.toml`: explicitly exclude skeleton subpackages from wheel.

## [0.0.1] - 2026-06-18

### Added

- Minimal PyPI placeholder.
- Initial `Calculator` skeleton and ctypes binding scaffold.
- PyPI publish workflow (GitHub Release triggered, trusted publishing).

[0.2.0]: https://github.com/kYangLi/aimspy/releases/tag/v0.2.0
[0.1.0]: https://github.com/kYangLi/aimspy/releases/tag/v0.1.0
[0.0.2]: https://github.com/kYangLi/aimspy/releases/tag/v0.0.2
[0.0.1]: https://github.com/kYangLi/aimspy/releases/tag/v0.0.1
