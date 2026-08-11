# Changelog

All notable changes to **aimspy** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`DeepHData`: optional `electric_response.h5` (dH/de) export** —
  electric-response first-order Hamiltonian (DFPT) support mirroring the
  existing Hamiltonian warmstart path. New `first_order_hamiltonian_entries` /
  `_fo_chunk_boundaries` / `_fo_chunk_shapes` dataclass fields;
  `first_order_hamiltonian=` keyword argument on `from_aimspy` /
  `from_memory` (accepts list of 3 `AimspyMatrix` `[x, y, z]` in Hartree);
  `set_first_order_hamiltonian` / `save_first_order_hamiltonian` /
  `to_first_order_aimspy` methods; auto read/write in `from_directory` /
  `save`. The `electric_response.h5` file uses the same `atom_pairs` as
  `hamiltonian.h5` but `chunk_shapes` rows are 3× (one block per Cartesian
  direction `[y, z, x]` = real spherical harmonics `m = -1, 0, +1`) and
  `entries` is 3× longer. Units are eV (converted from Hartree).
- **`Calculator`: DFPT electric-response capture + warmstart** —
  `CalculatorConfig.capture_first_order_hamiltonian: bool` flag;
  `Calculator.first_order_hamiltonian` property (list of 3 `AimspyMatrix`
  `[x, y, z]`); `Calculator.modify_init_first_order_ham(source=, strategy=)`
  method (direct + deferred mode, REPLACE/ADD strategies) that injects
  predicted dH/de before the initial U1 computation in `DFPT_cpscf`,
  accelerating CPSCF convergence (tested: 11→4 iterations on MoS2).
  Two new callbacks: `export_dHde` (post-CPSCF) and `modify_dHde`
  (pre-CPSCF).
- **`ExternalFirstOrderMatrixSource` Protocol** — structural typing
  protocol for first-order Hamiltonian sources accepted by
  `modify_init_first_order_ham`; implemented by `DeepHData`.
- `tests/unit/test_deeph_data.py` — 21 new unit tests for first-order
  functionality (roundtrip, direction order, shape validation, save/load,
  error cases, protocol, calculator config).
- `tests/unit/test_protocol_enum.py` — updated for 2 new callback names.

- **`DeepHData`: optional `force.h5` MD-style export** — new `force` /
  `energy_eV` dataclass fields; `force=` / `energy=` keyword arguments on
  `from_aimspy` / `from_memory` (accepts aims-order force array + Hartree
  energy, auto-reorders / converts); `set_force` / `save_force` methods; auto
  read/write in `from_directory` / `save`. The `force.h5` file uses a different
  layout from matrix `.h5` files: `cell` (3,3), `energy` scalar, `force`
  (n_atoms,3), `stress` (6,) zeros placeholder, with `formula` / `natoms`
  root attributes. Forces are in eV/Å (matching `calc.forces`); energy is in
  eV (converted from `calc.energy` Hartree).
- `tests/unit/test_deeph_data.py` — 16 new unit tests for force functionality
  (roundtrip, reorder, shape validation, list input, error cases).
- Integration tests: `test_baseline.py` now checks forces shape/finite;
  `test_export_deeph.py` adds 10 force.h5 cross-validation checks;
  `test_regression.py` adds 10 force/energy checks (total 60 checks).
- `tests/unit/test_force_close.py` — 9 unit tests for `force_close()` and
  `CalcState` transitions (no MPI/libaims required).
- `tests/conftest.py` — prevents pytest from collecting integration test
  scripts (which require MPI).
- `examples/continue_calc/run.py` — warmstart example using DeepH data
  produced by `from_scratch/run.py`.

### Known Limitations

- **DFPT serial mode** — `electric_field_serial .true.` (the default) is
  not supported for dH/de warmstart; only full-memory mode
  (`electric_field_serial .false.`) works correctly. Serial mode injects
  correctly but CPSCF converges slower (22 vs 16 iterations on MoS2) due
  to an unresolved interaction between the injected H1 and
  `evaluate_U1_electric_scalapack`. The `modify_dHde` callback skips
  injection when `n_dir != 3 or j_coord != 0` (serial mode signature).
- Makefile targets: `test-baseline`, `test-export-deeph`, `test-warmstart`,
  `test-capture-overlap`, `test-regression`, `test-strategies`,
  `test-integration`, `test-all`, `run-from-scratch`, `run-continue-calc`,
  `run-example`.
- `pyproject.toml`: `[tool.pytest.ini_options]` with `testpaths`.
- `pyproject.toml`: 4 new classifiers (License GPLv3+, OS Linux, Audience
  Science/Research, Python 3 :: Only).
- `.gitignore`: `*.out`, `*.h5`, `tests/data/MoS2/deeph_out/`,
  `tests/data/MoS2/_regression_*/`, `examples/*/deeph_data/`.
- Integration tests now exit with code 1 on failure (previously always 0).

### Changed

- **CLI**: `aimspy patch --version` renamed to `--patch-version` to avoid
  collision with Click's built-in `--version` flag.
- `forces` property no longer has a state guard — returns `None` when
  unavailable (before `calc()`, after `close()`, or `compute_forces` not
  set) instead of raising `AimspyStateError`.
- `logging.basicConfig()` at import time replaced with `NullHandler` —
  aimspy no longer configures the root logger.
- `Makefile build` target now produces both sdist and wheel (matches the
  `publish.yaml` workflow).
- `Makefile test` target now runs unit tests only (`pytest -v`);
  integration tests run via `make test-integration`.
- `Makefile clean` target now cleans test/example generated artifacts.
- `AIMSPY_TEST_NPROC` default unified to 8 for both tests and examples.
- README: `CalculatorConfig` types updated to `Path | str`; rank/opt-in
  descriptions for `hamiltonian`/`overlap`/`initial_hamiltonian` clarified;
  16 previously-undocumented public exports added to "Other public symbols".

### Fixed

- `DeepHData.from_memory`: empty dict `{}` for `hamiltonian_blocks`,
  `overlap_blocks`, or `initial_hamiltonian_blocks` now correctly produces
  `None` entries instead of mis-storing data or filling zeros.
- `DeepHData.save_hamiltonian` / `save_overlap` / `save_initial_hamiltonian`:
  now raise `AimspyConfigError` (was `ValueError`) — completes the
  standardization claimed in v0.2.0.
- `libloader.py`: MPI CDLL now anchored at module level to prevent GC
  from `dlclose`-ing it and removing RTLD_GLOBAL symbols.
- `export_ovlp` / `export_h0` callback wrappers now set
  `writeable=False` on the numpy view to protect Fortran `intent(in)`
  arrays from accidental modification.
- `AimspyInfo.from_c` renamed to `_from_c` (internal API, was
  incorrectly public).
- Removed dead `CallbackSpec` fields: `property_name`, `property_doc`,
  `raw_value_key`.
- `tests/test_strategies.py` / `test_regression.py` / `test_warmstart.py`:
  `deeph_warm` → `deeph_out` (use live-generated data, not stale reference).
- `tests/test_export_deeph.py`: removed DeepH-vs-DeepH comparison against
  non-existent `deeph_warm/` reference; replaced with cross-validation
  against in-memory matrices and `rs_hamiltonian.out`.
- CLI exception handling broadened to catch `OSError` /
  `subprocess.SubprocessError` (was only `KeyError` / `RuntimeError`).

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
- Unit test suite in `tests/unit/` (66 tests, no MPI/libaims required):
  `test_structure`, `test_protocol_enum`, `test_poscar`, `test_deeph_data`,
  `test_force_close`.
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
- State guards added to `info`, `structure`, `overlap` properties.
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
