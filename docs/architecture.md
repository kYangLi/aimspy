# Architecture Reference

This document is the maintainer-facing map of the aimspy system: how the
two repositories fit together, the complete C ABI, the callback
framework, runtime lifecycles, the conventions that keep Fortran and
Python consistent, and the known limitations. It distills the
full-codebase reviews behind the v0.2.1 basis-export release; line
references are stable as of that release but may drift — treat the code
as authoritative.

## 1. System overview: two repositories, one system

| Repository | Role |
|---|---|
| `pyapi` (this package) | Pure-Python ctypes driver: `Calculator` lifecycle, callback framework, data model, DeepH interface, CLI |
| `FHI-aims-deeph` | FHI-aims plus `src/aimspy_api/` — the in-process API layer the Python side binds to |

The bridge is the **bundled patch** (`aimspy/_patches/`, currently
`v0.2.1`): a cumulative `git diff dev..feature/aimspy` of the Fortran
repo that injects `src/aimspy_api/` plus trigger hooks into a stock
FHI-aims tree. The patch is regenerated from the Fortran repo via the
recipe stamped in the `Makefile` hunk it carries; it excludes
`src/aimspy_api/pytests`. **Consequence: Fortran-source changes require
regenerating the patch** (and, during development, may reuse the same
version stamp — see §11).

`aimspy_init` / `aimspy_run` / `aimspy_finalize` wrap
`aims_initialize` / `aims_run` / `aims_finalize`; everything else is
 callbacks and pull-style accessors around module-level Fortran state.

## 2. Layered architecture (Python side)

```
Tools      cli.py (patch)  _cli_viz.py (viz-basis / viz-grid)  viz.py  viz_basis.py
Interface  interface/__init__.py (Protocols)  interface/deeph/data.py (DeepHData)
Core       calculator.py  — state machine, lifecycle, Strategy, warmstart
Data       data.py (AimspyInfo, CsrMatrixDescriptor)  info.py  matrix.py
           structure.py  grid_data.py  basis_data.py
Callbacks  _callbacks/registry.py (9 specs)  _callbacks/base.py (manager + wrappers)
Binding    _binding/libloader.py  prototypes.py (18 symbols)  ctypes_types.py  callback_types.py
                    │ ctypes.CDLL(libaims.so, RTLD_GLOBAL)
Fortran    aimspy_api/ (main, api_bank, callback, register, info, export_grid_data,
           export_basis_data)  → trigger points in aims proper (§6)
```

Design principles worth knowing before touching anything:

- **Exception firewall**: a raising Python callback never propagates
  into the Fortran call stack. Wrappers catch everything, record
  `(name, exc, tb)` triples, and the Calculator re-raises them as one
  `AimspyCallbackError` (`.callback_errors`) at the end of `init()` or
  `calc()`. Forces are harvested *before* the check so they survive
  callback failures.
- **Old-library tolerance**: `BindingLib` resolves symbols lazily;
  missing symbols (older libaims) are skipped at setup time and raise
  `AimspyBindingError` only if actually used.
- **Everything escaping to the user is a copy** (§8) — the only
  exceptions are the deliberately short-lived views inside `modify_*`
  callbacks.
- **Extension recipe**: adding a callback touches 5 places (Fortran
  patch, `callback_types.py`, `prototypes.py`, `registry.py`, and the
  wrapper branch in `_callbacks/base.py`); both `base.py` and
  `registry.py` state this in their headers.

## 3. The C ABI surface (18 symbols)

| Group | Symbols |
|---|---|
| Lifecycle | `aimspy_init(comm, logfile)`, `aimspy_run()`, `aimspy_finalize()`, `aimspy_reset_callbacks()` |
| Pull accessors | `aimspy_get_info()` → struct ptr; `c_rs_hamiltonian()` / `c_rs_overlap()` → rank-0 buffer ptrs; `aimspy_energy()` (by value); `aimspy_forces()` → ptr or NULL |
| Registration | 9 × `aimspy_register_<name>_callback(cb, aux[, extra_ptr])` |

Struct mirrors (`AimspyInfoC`, `CsrMxDescrC`, `GridDescrC`,
`BasisDescrC` in `ctypes_types.py`) are ABI contracts: field order must
match the Fortran `type, bind(C)` declarations byte-for-byte; char data
is passed as separate pointers to dodge padding mismatches. `libloader`
loads mpi4py's own shared library `RTLD_GLOBAL` into a module-level
anchor (GC would `dlclose` it otherwise and remove MPI symbols that
libaims resolves lazily); MPI must therefore be initialized (mpi4py
imported) before `load_aims_lib`.

## 4. The nine callbacks

| # | Name | Fires (Fortran) | When | Payload |
|---|---|---|---|---|
| 9 | `export_basis_data` | `prepare_scf.f90` after `shrink_fixed_basis_phi_thresh` | **inside `aimspy_init`** — must be registered pre-init | 3 spline arrays `(4, n_max_grid, n_basis_fns)` + per-species grid params |
| 1 | `get_descr` | `initialize_scf.f90` | first `calc()`, pre-SCF | CSR index arrays (descriptor) |
| 2 | `export_ovlp` | `initialize_scf.f90` | pre-SCF | flat S (n_spin arg hardcoded 1) |
| 3 | `export_h0` | `initialize_scf.f90` | pre-SCF | flat H0 `(n_ham, n_spin)` |
| 5 | `python_func` | `initialize_scf.f90` | pre-SCF, between export_h0 and modify_h0 | none (generic hook) |
| 4 | `modify_h0` | `initialize_scf.f90` | pre-SCF; writeable buffer + warmstart short-circuit afterwards | live H0 (inject via `memmove` inside the callback) |
| — | (vdW store) | `integrate_hamiltonian_matrix_p2.f90` | every SCF iteration, vdW runs | buffer only, not a callback |
| 8 | `export_grid_data` | `scf_solver.f90` post-loop | after each `scf_solver` call (per geometry), even if SCF did not converge | this rank's grid subset + fields |
| 7 | `modify_dHde` | `DFPT_module.f90` | pre-CPSCF, before initial U1 (periodic only) | raw address of `DFPT_first_order_H_sparse` |
| 6 | `export_dHde` | `DFPT_module.f90` | CPSCF convergence (periodic only) | flat dH/de `(n_dir, 1, n_ham, n_spin)` |

Registration rules enforced by `Calculator`:

- `export_basis_data` **must** be registered before `aimspy_init`
  (built-in via `capture_basis_data=True`; a user pre-init registration
  replaces the built-in and `calc.basis_data` stays None unless the
  user stores it). Post-init registration warns "already fired".
- All others register any time before `calc()`; user registrations
  take precedence over the built-in wiring (`_wire_callbacks` skips a
  name if already registered).
- UNINIT registrations are deferred and applied inside `init()`;
  FAILED/FINALIZED raise `AimspyStateError`; DONE warns "never fire".
- **Every rank must register the same callbacks** — trigger sites have
  no rank guards (except the rank-0 `c_*` snapshots).

## 5. Calculator state machine

```
UNINIT ──init()──▶ INITED ──calc()──▶ RUNNING ──ok──▶ DONE
   │                 │                    │
   │                 │                    └──error──▶ FAILED ──close()──▶ FINALIZED
   │                 └──close()──▶ FINALIZED
   └──init() error──▶ (FAILED →) FINALIZED, exception re-raised
```

- double `init()` raises; `calc()` requires INITED (no re-run after
  DONE); `close()` in RUNNING raises (use `force_close()`);
  `close()`/`force_close()` are idempotent and never raise.
- `_clear_all_state` ordering is **load-bearing**: Fortran
  `aimspy_reset_callbacks` must run *before* `_cb_mgr.clear()` — after
  clear(), the raw `id(aux)` pointers passed to Fortran dangle.
- The supported multi-geometry pattern is **one full
  init/run/finalize cycle per geometry** (see §6d).

## 6. Execution timelines

**(a) init-only basis capture** (`capture_basis_data=True`, no
`calc()`): `aimspy_init` → `prepare_scf` fires `export_basis_data` →
info/structure fetch → callbacks wired → close. SCF never runs; ELSI
never initializes (guarded by `elsi_scf_ready`, which is what makes
this workflow survivable at all).

**(b) standard SCF + capture**: `calc()` → `initialize_scf`
(get_descr → export_ovlp → export_h0 → python_func → modify_h0 →
[with modify_h0: warmstart short-circuit via `advance_KS_solution`,
`restart_zero_iteration=.true.`]) → SCF loop (rank-0 refreshes
`c_hamiltonian` every iteration) → post-loop `export_grid_data` →
Python pulls matrices/energy/forces.

**(c) DFPT dH/de**: SCF as (b), then `linear_response_wrapper` →
`elecres_calculation`. Full-memory mode: one CPSCF with all 3
directions (`n_dir=3, j_coord=0`), `modify_dHde` fires once, then
`export_dHde` once. Serial mode: three sequential CPSCF solves
(`n_dir=1, j_coord=1/2/3`), callbacks fire per direction; Python
accumulates the three exports. Periodic systems only.

**(d) multi-geometry**: in-process relaxation re-enters
`reinitialize_scf`, which **does not re-fire** the matrix callbacks and
leaves the rank-0 `c_overlap` stale — the supported pattern is a fresh
Calculator per geometry. Calling `aimspy_run` twice on one init is not
a geometry-update mechanism (post-processing only on the second call).

## 7. Conventions (the correctness-critical table)

| Dimension | Convention |
|---|---|
| Units | in-memory matrices Hartree; DeepH disk eV; coords/lattice Å; k-points & recip lattice 1/Å; forces eV/Å; GridData stays in aims-native units (bohr, Hartree, e/bohr³) |
| R vectors | `R_aimspy = R_deeph = −R_aims` — DeepH↔aimspy never flips; CSR↔block conversion flips |
| Atom order | aimspy blocks use aims order; DeepH/POSCAR uses element-grouped order; all conversions go through `AimspyStructure.build_atom_permutation()` |
| Orbital parity | wiki/DeepH parity (−1 iff m>0 and m odd), self-inverse, applied in the matrix layer |
| dH/de directions | DeepH order `[y, z, x]` (real spherical harmonics m = −1, 0, +1) ↔ Cartesian `[x, y, z]` |
| Index base | Fortran 1-based → Python 0-based everywhere (`_view_i(...)` − 1 at the boundary); GridData index arrays likewise |
| Spin | `n_spin = 1` only — enforced at CSR conversion, DeepH read, and DeepH write |
| Splines | integer grid-index axis: `i_r = 1 + ln(r/r_grid_min)/ln(r_grid_inc)`, snap ±1e-10, clip `[1, n_g−1]`, Horner `c1 + t(c2 + t(c3 + t·c4))`, zero outside `[r_grid_min, outer_radius]` |
| GridData `rho0` | free-atom density; the 4π factor of aims' `free_rho_superpos` is removed at import; `delta_rho` per spin channel references `0.5·rho_free` |
| `vks` | scalar part only (exact for LDA; GGA's `4·xc_gradient_deriv` vector term is not exported) |

## 8. Memory ownership rules

| Object | Owner | Valid after callback / finalize |
|---|---|---|
| `AimspyInfo`, `CsrMatrixDescriptor`, `GridData`, `BasisData`, captured `AimspyMatrix`, dH/de copies, forces | Python (copies at capture) | yes — but the Calculator drops its references at `close()`; keep your own |
| `export_ovlp` / `export_h0` views | Fortran | no — read-only views die when the callback returns; the built-in handlers convert to blocks immediately |
| `modify_h0` / `modify_dHde` buffers | Fortran (`intent(inout)`) | inject via `memmove` *inside* the callback; the raw address is valid only then |
| aux objects | Python (`CallbackManager._auxs` holds the strong refs; Fortran gets `id(aux)`) | until `clear()` — hence the reset-before-clear ordering |

RSS flatness across sequential Calculators relies on: every export
module deallocating its buffers in `aimspy_finalize`, `aimspy_run`
deallocating `c_hamiltonian`/`c_overlap` at entry, and the reference
-cycle hygiene (deferred modify closures never capture `self`;
wrappers capture the error list, not the manager; `_clear_all_state`
nulls SimpleNamespace internals).

## 9. Fortran global state (api_bank)

- `aimspy_initialized` — set before `aims_initialize`, **never reset**;
  after finalize the trigger blocks still run but every `*_func`
  early-returns on null funptrs (see §11).
- `c_hamiltonian` / `c_overlap` — rank-0 snapshots; deallocated at each
  `aimspy_run` entry so the next cycle re-allocates cleanly.
- `aimspy_callback_hd` — the single handle holding all 9 funptrs +
  aux pointers + registration flags; `reset_all` (from
  `aimspy_finalize` and `aimspy_reset_callbacks`) nulls everything.
- `elsi_scf_ready` — guards `elsi_finalize` in `aims_finalize` for
  init-only workflows (ELSI's `elsi_stop` ends in a bare Fortran
  `stop`, which would kill the interpreter).
- Logfile unit 20 — defensively closed before `open` in `aimspy_init`
  and after `aims_finalize` (iostat-guarded); without this, a second
  Calculator reusing the same logfile path crashed with forrtl
  severe(104).

## 10. Test infrastructure

- **Unit** (`tests/unit`, `testpaths`): no MPI, no libaims, seconds.
  MPI-dependent importers (grid gather) need `mpi4py`'s libmpi —
  source the MPI environment first or 4 gather tests fail on import
  (environmental, not regression).
- **Integration** (`tests/*.py`, 16 scripts, via `make
  test-integration`): every script aborts at import without
  `AIMSPY_TEST_AIMS_LIBPATH` — hence `tests/conftest.py`
  `collect_ignore` must list every new one. Dependency chain:
  baseline → export-deeph → warmstart/regression/strategies;
  dHde-capture → dHde-inject-{direct,defer}; dHde-serial-capture →
  dHde-serial-inject; LDA trio (grid, basis×2) and callback-reset /
  memory-loop independent.
- Strategy variants run in **separate sub-MPI processes** (FHI-aims is
  a global Fortran singleton — one init/finalize per process).

## 11. Known limitations (deliberate, with reasons)

| Limitation | Status / reason |
|---|---|
| Patch version-stamp reuse during development (v0.2.1 content regenerated in place) | accepted while pre-release; revisit at v0.2.2 (uninstall/upgrade from an old v0.2.1 tree would fail) |
| `save_h5` mid-write failure leaves written groups absent from `species_list`, and re-runs skip them | accepted (dev-stage; disk-full scenario) |
| Same-element multi-species (`S` + `S1` tags) — the second species' basis is silently not exported to `basis.h5` (groups keyed by bare element symbol) | accepted; disambiguation (e.g. `S_2` group names) deferred |
| Spline kernel duplicated in three places (`BasisData._evaluate_spline_fn`, `evaluate_du_dr`, `viz_basis._evaluate_spline`) | minimal-invasiveness decision; all three verified in sync against Fortran `val_spline` |
| `aimspy_initialized` never reset in `aimspy_finalize` | harmless today (null-funptr early returns); matters only if aimspy and non-aimspy drivers share one process |
| `c_hamiltonian`/`c_overlap` not freed by `aimspy_finalize` (freed at next `aimspy_run` entry) | bounded one-set retention on rank 0 |
| BSSE re-shrink path does not re-fire `export_basis_data` | corner case (BSSE + basis capture combined) |
| x2c/q4c small-component splines not exported (large component only) | documentation-level limitation |
| Spline export buffers replicated on every rank (no root-only path) | deliberate — Python registers/captures on all ranks |
| Rank-divergent callback registration → MPI hang | inherent to the all-ranks trigger design; inherited pattern |
| In-process relaxation: matrix callbacks fire on the first geometry only; `c_overlap` stale from step 2 | supported pattern is one Calculator per geometry |
| `AimspyMatrix` / DeepH path is `n_spin=1` only | spin-polarized support not implemented |
| dH/de callbacks require periodic systems (`n_periodic > 0`) | aperiodic electric response uses dense H1 and never triggers them |
| `export_ovlp` hardcodes `n_spin=1` in the callback argument | consistent with the n_spin=1 limitation |
| matplotlib/scipy undeclared in core dependencies | available via the `[viz]` extra; CLI fails with an install hint |
| Restart file pre-empts warmstart injection (`keep_restart_info` + existing restart) | aims-native behavior, documented in initialize_scf |

## 12. Where to look for what

| Topic | File |
|---|---|
| Lifecycle, state machine, warmstart strategies | `aimspy/calculator.py` |
| Callback catalogue (authoritative) | `aimspy/_callbacks/registry.py` |
| Callback wrappers / buffer conversion / exception firewall | `aimspy/_callbacks/base.py` |
| ABI: symbols, structs, CFUNCTYPEs | `aimspy/_binding/{prototypes,ctypes_types,callback_types}.py` |
| CSR ↔ block-sparse conversion, conventions | `aimspy/matrix.py`, `aimspy/structure.py` |
| DeepH on-disk format, atom reordering, unit flips | `aimspy/interface/deeph/data.py` |
| Patch apply/uninstall machinery | `aimspy/_patches/_apply.py` |
| Fortran entry points / global state | `FHI-aims-deeph/src/aimspy_api/{main,api_bank}.f90` |
| Fortran dispatch / descriptor structs | `FHI-aims-deeph/src/aimspy_api/callback.f90` |
| Export buffer assemblies | `.../export_grid_data.f90`, `.../export_basis_data.f90` |
