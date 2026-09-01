# Changelog

All notable changes to **aimspy** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-09-01

### Added

- **Real-space grid data capture (`export_grid_data`)** — new callback
  (8th) exporting converged density, Kohn-Sham potential, Hartree
  potential, and grid geometry (coords/weights/indices) after SCF.
  Includes vdW potential in `vks` when enabled in FHI-aims
  (`use_vdw_correction_hirshfeld_sc` / `use_mbd_std` / `use_libmbd`).
  `GridData` dataclass with derived quantities (`delta_rho`, `delta_vks`,
  `vxc`, `rho_free`), npz I/O, and Gatherv-based MPI gather (root peak
  memory ~1x vs ~3x for pickle-based gather).
- **Visualization helpers (`aimspy.viz`)** — `scatter_slice`,
  `slice_contour`, `radial_profile`, `isosurface` for grid data analysis.
  `radial_profile` uses exact nuclear positions from `atom_coords` when
  available (fallback: grid-point centroid).
- `tests/unit/test_grid_data.py` (19 tests), `tests/unit/test_viz.py`
  (20 tests), `tests/test_grid_data_capture.py` (integration test).
- `tests/data/MoS2_LDA/` — LDA test fixture for grid data capture.
- **NAO radial basis capture (`export_basis_data`)** — new callback
  (9th) exporting the full cubic-spline representation of all NAO radial
  basis functions (u(r), (e−v)·u(r), du/dr) plus per-species logarithmic
  grid parameters and outer radii, fired once inside `prepare_scf`
  (before SCF). `BasisData` dataclass with spline evaluation —
  `evaluate_u` / `evaluate_phi` / `evaluate_du_dr` /
  `evaluate_kinetic` / `evaluate_deriv` (per-function species map
  attached automatically at init; `evaluate_deriv` reads the
  aims-native `spline_deriv`, non-zero only when `use_basis_gradients`
  is active) — and incremental element-per-group `basis.h5` export for
  building reusable basis libraries.
  Registered before `aimspy_init` (`capture_basis_data=True`).
- **Basis visualization (`aimspy.viz_basis`) + CLI** — `plot_radial_basis`
  plots u(r) or φ(r) from a `basis.h5` file (runtime-free), with
  per-l panels, optional log-x (evenly spreads the log-grid sampling),
  and log-grid rug markers.  New CLI commands
  `aimspy viz-basis` (all elements, `-j` parallel) and
  `aimspy viz-grid` (scatter/contour/radial over `GridData` npz).
- `tests/unit/test_basis_data.py` (22 tests),
  `tests/unit/test_viz_basis.py` (39 tests),
  `tests/test_basis_export.py` (integration test, 35 checks),
  `tests/test_basis_callback_paths.py` (integration test: pre-init
  registration / init-time error surfacing / user-precedence).
- `tests/unit/test_viz_basis.py::TestCLIErrorPaths` — CLI error-path
  regression tests (corrupt h5/npz, missing datasets, NaN/negative
  `--r-max`, bad `-o` paths).
- **`[viz]` optional dependency extra** — `pip install aimspy[viz]`
  installs matplotlib + scipy for the visualization helpers; the CLI
  commands now fail with an install hint instead of a raw
  `ModuleNotFoundError` when they are missing.
- `docs/architecture.md` — full architecture reference (layered
  design, ABI/callback inventory, state machine, execution timelines,
  conventions, memory-ownership rules, known limitations).
- Makefile targets `test-grid-data-capture`, `test-basis-export`,
  `test-basis-callback-paths` (previously runnable only by hand);
  `test-integration` now includes them.

### Changed

- **Fortran buffer lifecycle** — `aimspy_export_grid_data_finalize()`
  explicitly deallocates all 8 module-level buffers (coords, partition_tab,
  indices, vks, vks0, c_vdw_potential) in `aimspy_finalize`, preventing
  ~500 MB retention after Calculator close.
- **Basis export guards** — `aimspy_export_basis_data_run` is now only
  called when `aimspy_initialized` (plain aims runs no longer allocate the
  ~MB export buffers), and early-returns when the `export_basis_data`
  callback was never registered (new `export_basis_data_registered` flag,
  same pattern as `modify_h0_registered`).
- **viz defaults** — `scatter_slice` marker size `s` 2.0 → 5.0,
  `slice_contour` `levels` 50 → 60 (denser atom-centred grids stay
  readable).  The new defaults are now pinned by unit tests.
- **Docs refresh** — `key_concepts.md` patch-system section now
  describes all three bundled versions (latest `v0.2.1`), the callback
  wiring table covers all 8 config-driven callbacks (grid/dHde rows
  added), and the trigger-point list includes `DFPT_module.f90`;
  `api_reference.rst` preamble clarifies the import path of the viz
  helpers; README Core Features gains the NAO basis capture entry and
  an `[viz]` install note.
- **`register_callback('export_basis_data', ...)` timing** — the
  registration semantics are: a pre-init registration is applied *before*
  `aimspy_init` (the callback fires inside `aimspy_init` itself, so this
  is the only point at which it can take effect).  Registering after
  `init()` issues a `UserWarning` — by then the callback has already
  fired and would never be called.
- **Init-time callback errors** — exceptions raised inside callbacks that
  fire during `aimspy_init` (`export_basis_data`) now surface as
  `AimspyCallbackError` from `init()` itself, instead of being deferred
  to the first `calc()` call.

### Fixed

- **CLI tracebacks on malformed input** — `aimspy viz-basis` /
  `aimspy viz-grid` now wrap their entire command bodies (including
  file loading, plotting, and figure saving) and convert library-level
  exceptions (`ValueError`/`IndexError`/`KeyError`/`OSError`/`TypeError`
  /`EOFError`/`BadZipFile`) into clean `Error:` messages instead of raw
  tracebacks.  Previously e.g. a corrupt `basis.h5` or truncated `.npz`
  crashed with a full traceback, the interactive (no `-o`) path of
  `viz-basis` had no wrapping at all, and `viz-basis` did not catch
  `KeyError`/`OSError`.  `--r-max nan` is now rejected (`r_max must be
  a positive finite number`) instead of silently producing an empty
  plot.
- **`aimspy_finalize` on a never-initialized runtime** — two pre-init
  failure paths (older libaims lacking
  `aimspy_register_export_basis_data_callback` + `capture_basis_data`,
  and an invalid pending callback name) reached the init()-error
  handler's `_defensive_finalize`, calling `aimspy_finalize` on a
  runtime whose `aimspy_init` never ran.  init() now tracks whether
  `aimspy_init` was actually entered and skips the Fortran-side
  finalize when it was not (Python-side callback reset still runs).
- **`register_callback` in FAILED state** — the docstring promised an
  `AimspyStateError`, but registration silently succeeded on the still
  live callback manager.  FAILED now raises (matching FINALIZED); the
  previous operation aborted, so registration could never be useful.
- **`save_h5` species_list duplication on h5py 3.0.x** — the
  read-back of the `species_list` attribute did not decode `bytes`
  entries (returned by h5py 3.0.x for vlen-string attrs), making the
  membership test always True and accumulating duplicate entries on
  every save.  Entries are now decoded before the test; the empty
  initializer is also written with an explicit string dtype (a bare
  `[]` was stored as an empty float64 attribute).
- **`save_h5` contiguity guard** — the per-species function-range
  computation silently assumed species-major ordering of
  `basisfn_species`; an interleaved ordering would have exported
  another species' functions under the wrong element.  A loud
  `ValueError` now fires on non-contiguous indices.
- `tests/conftest.py` `collect_ignore` now lists
  `test_basis_export.py` / `test_basis_callback_paths.py` — collecting
  `tests/` directly previously aborted the whole pytest run at import
  (`comm.Abort(1)` when `AIMSPY_TEST_AIMS_LIBPATH` is unset).
- Vacuous `test_rug_respects_r_max` unit test now actually exercises
  the rug filter (bohr units + r_max below the outer grid points);
  zeta numbering is now tested through the real `save_h5` path with
  duplicate (n, l) pairs (previously only a local re-implementation
  was tested).
- **Process crash on second `aimspy_init` reusing the same logfile
  (forrtl severe 104)** — `aimspy_init` opened the logfile (unit 20)
  with `status='replace'` but nothing ever closed the unit; a second
  `Calculator` in the same process pointing at the *same* logfile path
  hit "incorrect STATUS= specifier value for connected file" and died
  with a native crash.  `aimspy_init` now defensively closes the unit
  before opening, and `aimspy_finalize` closes it after
  `aims_finalize` (iostat-guarded, idempotent).
  All existing multi-Calculator tests unknowingly avoided this by
  using a distinct logfile per cycle.
- **Process death at close() for init-only workflows (ELSI)** —
  `final_deallocations` unconditionally called `elsi_finalize(eh_scf)`
  even when no SCF ran (`eh_scf` never initialized), and ELSI's
  `elsi_stop` ends with a bare Fortran `stop` — silently killing the
  whole Python process (exit code 0) at `Calculator.close()`.
  `aims_elsi_finalize_scf` is now guarded by an `elsi_scf_ready` flag
  set in `aims_elsi_init_scf` and cleared after finalization.
  This affected the `capture_basis_data` init-only pattern and any
  init-failure cleanup path; `tests/test_basis_export.py` previously
  lost its final "ALL CHECKS PASSED" line to this.
- **vdW potential omission** — `vks` now includes vdW correction when
  vdW is active. Previously `vks = V_H + v_xc` only, missing the vdW
  contribution that `integrate_hamiltonian_matrix_p2` adds to the
  Hamiltonian.

- **Fortran callback deregistration + `aimspy_reset_callbacks`** — new
  `TAimspyCallback.reset_all` clears all registered funptrs / aux pointers /
  input pointers / registered-flags; called inside `aimspy_finalize` and
  exposed via the new `aimspy_reset_callbacks` bind(c) entry point, which
  `Calculator.close()` / `force_close()` invoke (when the symbol exists).
  Prevents a second `Calculator` in the same process from calling a
  dangling Python function pointer left over from a previous `Calculator`.
- **Serial DFPT dH/de support** — `export_dHde` now accumulates the three
  Cartesian directions across the three serial CPSCF calls
  (`n_dir=1, j_coord∈{1,2,3}`); `modify_dHde` injects only the current
  direction in serial mode. Previously serial mode was silently ignored.
- `tests/test_dHde_serial_capture.py` — self-contained serial capture test:
  runs full-memory (reference) + serial capture and cross-validates the
  dH/de tensors to within the CPSCF convergence noise (global relative
  difference < 1e-4; the two modes run independent CPSCF cycles converging
  to `dfpt_sc_accuracy_dm` ~1e-3, so they do not agree to machine precision).
- `tests/test_dHde_serial_inject.py` — serial warmstart test: injects the
  serial capture product and verifies per-direction CPSCF iteration counts
  are reduced (MoS2: [11,12,12] → [2,2,4]).
- `tests/data/MoS2_DFPT_serial/` — serial DFPT test data
  (`electric_field_serial .true.`).
- `tests/test_callback_reset.py` — same-process multi-Calculator test
  verifying no stale-callback invocation after close.

### Changed (breaking)

- **Spin-polarized (n_spin=2) data now raises instead of silently reading
  spin channel 0.** `AimspyMatrix.from_aims_csr` / `to_aims_csr` raise
  `AimspyError` when `csr_descr.n_spin != 1`; `DeepHData.from_directory`
  raises `AimspyConfigError` when `info.json` has `spinful: true`.
  Previously spin channel 1 was silently discarded.
- **Callback failure now marks the Calculator as FAILED** (was DONE),
  so DONE-only properties (`hamiltonian`, `energy`, …) are inaccessible
  after a callback error — results are untrustworthy.
- `modify_init_ham` / `modify_init_first_order_ham` now raise
  `AimspyStateError` if called after `init()` / `do()` (previously a
  silent no-op).

### Fixed

- **`dHde_warmstart_serial.py` buffer layout** — the flat injection buffer
  is now built as contiguous per-direction chunks (C-order ravel of the
  transposed array); the previous Fortran-order ravel interleaved the
  directions, injecting wrong data in serial mode.
- **`rs_matrix.py` CSR descriptor ABI** — added the missing
  `n_cells_array` field to `AimsCsrMxDescr.c_struct`, fixing a struct
  layout mismatch with the Fortran `TAimspyCsrMxDescr`.
- `DeepHData.from_directory` now validates `atom_pairs` consistency across
  matrix files, reordering per-pair when the same set is stored in a
  different order and raising when the pair sets differ.
- `DeepHData.from_memory` / `set_first_order_hamiltonian` require all
  three dH/de directions `[x, y, z]` to be non-empty (dH/de is only
  meaningful when all three are present).
- `save_first_order_hamiltonian` raises a clear error when the first-order
  chunk layout is not set.
- `modify_dHde` validates the `to_aims_csr` output shape before memmove.
- Fortran: `c_loc` calls in `fill_mx_descr` are now guarded by
  `allocated()`; `c_f_string` guards against a NULL C pointer; the five
  matrix dummies in the callback entry points are declared `contiguous`.
- `Calculator.hamiltonian` is now cached after the first access.
- Failed `calc()` / `init()` now release large matrices retained in the
  runtime aux dict (overlap, initial/converged/first-order Hamiltonian,
  external sources).

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

[0.2.1]: https://github.com/kYangLi/aimspy/releases/tag/v0.2.1
[0.2.0]: https://github.com/kYangLi/aimspy/releases/tag/v0.2.0
[0.1.0]: https://github.com/kYangLi/aimspy/releases/tag/v0.1.0
[0.0.2]: https://github.com/kYangLi/aimspy/releases/tag/v0.0.2
[0.0.1]: https://github.com/kYangLi/aimspy/releases/tag/v0.0.1
