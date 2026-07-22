# Key Concepts

This section describes the core architecture and data formats used by AimsPy. Understanding these concepts is essential for using the API effectively and for extending the package with new functionality.

## Overview

AimsPy drives [FHI-aims](https://aims-code.rg.mpg.de/) DFT calculations directly from Python by loading a *patched* `libaims.so` via `ctypes`. There is no subprocess and no file-staged I/O on the hot path — Hamiltonian, overlap, energy, and forces are exchanged as in-memory arrays through a callback framework exposed by the bundled FHI-aims patch.

The package is organised in three layers:

| Layer | Purpose | Public? |
|-------|---------|---------|
| `aimspy.calculator`, `aimspy.matrix`, `aimspy.structure`, `aimspy.interface.deeph` | User-facing API | Yes |
| `aimspy._callbacks` | Callback framework (`CallbackManager`, 5 hook-point specs) | No |
| `aimspy._binding`, `aimspy._patches` | ctypes binding to `libaims.so`; versioned FHI-aims diffs | No |

The central user-facing class is `Calculator`, whose lifecycle is governed by the `CalcState` state machine.

## In-Memory Architecture

### The ctypes binding layer

`aimspy._binding.libloader.load_aims_lib(lib_path)` is the **only** place that calls `ctypes.CDLL` on `libaims.so`. Two details matter:

1. **MPICH symbol visibility.** `mpi4py`'s own shared object must be loaded with `RTLD_GLOBAL` *and anchored at module level* (`_mpi_cdll_anchor`). Without the module-level anchor, the `CDLL` would be garbage-collected (and `dlclose`'d) when the function returns, removing the global symbols that `libaims.so` needs for lazy MPI symbol resolution. AimsPy loads `libaims.so` itself with `RTLD_GLOBAL` as well.
2. **Forward-compatible symbol probing.** `BindingLib` (in `aimspy._binding.prototypes`) remembers which C symbols were detected and exposes a `has(name)` predicate. `setup_prototypes` silently skips symbols missing in older `libaims` builds, so a single AimsPy release can drive multiple patch versions.

### The `Calculator` lifecycle

`CalcState` is a six-state enum with a directed lifecycle:

```
UNINIT ──init()──> INITED ──calc()──> [RUNNING] ──> DONE
                     │                                   │
                     └──close()──────────────────────────┘
                                                │
              close()/force_close() ──────────> FINALIZED
   (any state, on error) ──> FAILED ──force_close()──> FINALIZED
```

| State | Meaning | Allowed next |
|-------|---------|--------------|
| `UNINIT` | Freshly constructed | `init` / `do` / `force_close` |
| `INITED` | `aimspy_init` done; callbacks registered; `info`/`structure` readable | `calc` / `close` |
| `RUNNING` | Transient inside `calc()` while `aimspy_run` is executing | `DONE` / `FAILED` |
| `DONE` | SCF converged; `hamiltonian`/`energy`/`forces` readable | `close` |
| `FAILED` | Operation aborted; Fortran runtime in unknown state | `force_close` only |
| `FINALIZED` | `aimspy_finalize` called; terminal | — |

State transitions are guarded — calling `calc()` from `UNINIT`, or `close()` from `RUNNING`, raises `AimspyStateError`. Use `force_close()` for safe recovery from any state.

> **Thread safety**: `Calculator` is **not thread-safe**. FHI-aims uses `chdir(2)` (process-global) internally, so concurrent `Calculator` operations in the same process will race. The typical MPI usage is one rank = one process.

## The 5 Callback Hook Points

The bundled FHI-aims patch inserts five trigger points inside `src/initialize_scf.f90`, after `reshape_matrices` and before the initial diagonalisation. They fire in this order:

| # | Callback | Register symbol | Purpose |
|---|----------|-----------------|---------|
| 1 | `get_descr` | `aimspy_register_get_descr_callback` | Fill the `TAimspyCsrMxDescr` struct (CSR layout) |
| 2 | `export_ovlp` | `aimspy_register_export_ovlp_callback` | Export the overlap matrix |
| 3 | `export_h0` | `aimspy_register_export_h0_callback` | Export the free-atom initial Hamiltonian `H_init` |
| 4 | `python_func` | `aimspy_register_python_callback` | Generic Python hook (deferred source generation) |
| 5 | `modify_h0` | `aimspy_register_modify_h0_callback` | Inject the modified `H_init` back into FHI-aims |

When `modify_h0` is registered, the patch **short-circuits** the standard initial diagonalisation: it calls `advance_KS_solution` directly on the injected Hamiltonian and sets `restart_zero_iteration=.true.`, which is what enables 1-iteration warmstart.

### How callbacks are wired

`CalculatorConfig` flags control which default callbacks are auto-registered by `Calculator._wire_callbacks`:

| Flag | Enables | Effect |
|------|---------|--------|
| (always) | `get_descr` | `calc.csr_descr` populated |
| `capture_overlap=True` | `export_ovlp` | `calc.overlap` returns live overlap on all ranks |
| `capture_initial_hamiltonian=True` | `export_h0` | `calc.initial_hamiltonian` populated |
| `modify_init_ham(...)` called | `python_func` + `modify_h0` | warmstart / scaling / custom modification |

Users can also register custom callbacks via `Calculator.register_callback(name, fn, aux, extra_ptr)` for advanced use cases.

### Error handling

A Python exception raised inside a callback never crashes Fortran. The `CallbackManager` records `(name, exception, traceback_str)` tuples and lets `calc()` complete. After `aimspy_run` returns, `Calculator._check_callback_errors` raises a single `AimspyCallbackError` aggregating all failures, with the per-callback details preserved on `exc.callback_errors`. Notably, **forces are captured before the callback error check**, so they survive even when a callback raises.

## Hamiltonian Modification Strategies

`Calculator.modify_init_ham(source=..., strategy=...)` configures how the live `H_init` buffer is mutated before SCF starts. Four strategies ship built-in, dispatched by the pure function `_apply_strategy`:

| Strategy | Behaviour | Required argument | Typical use |
|----------|-----------|-------------------|-------------|
| `REPLACE` | Clear and copy external blocks into the live `H_init` | `source=` | Warmstart with a DeepH prediction |
| `ADD` | Accumulate external blocks on top of the live `H_init` | `source=` | Perturbation / hybrid construction |
| `SCALE` | Multiply the live `H_init` by a constant factor | `factor=` (float) | Scaling experiments |
| `CUSTOM` | Call `custom_fn(live, external, structure, aux)` to mutate the live matrix in place | `custom_fn=` (callable) | Arbitrary transforms |

### Direct vs. deferred source

`modify_init_ham` supports two modes:

- **Direct mode** (`source=` is passed): the source object is stored immediately. The `python_func` callback then converts it via `source.to_aimspy(structure)` and stores the result in `_runtime_aux["external_aimspy"]`, which `modify_h0` reads.

- **Deferred mode** (used as a decorator): the user function is called *during* the `python_func` callback, after `export_h0` and `export_ovlp` have fired, so it has live access to `calculator.initial_hamiltonian` and `calculator.overlap`. This is essential when the external source needs the runtime structure to be built.

```python
# Direct
calc.modify_init_ham(source=data, strategy=Strategy.REPLACE)

# Deferred
@calc.modify_init_ham(strategy=Strategy.REPLACE, option={"path": "deeph_warm/"})
def gen_source(calculator, option):
    return DeepHData.from_directory(option["path"])
```

## ExternalMatrixSource Protocol

The `ExternalMatrixSource` `Protocol` (`aimspy.interface`) is the contract for any external matrix provider:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ExternalMatrixSource(Protocol):
    def to_aimspy(self, structure: AimspyStructure) -> AimspyMatrix: ...
```

The reference implementation is `DeepHData` (`aimspy.interface.deeph`), which reads the DeepH on-disk format (`POSCAR` + `info.json` + `.h5`) and converts to `AimspyMatrix`. Adding a new external format is a single subpackage under `aimspy/interface/<format>/` with a class satisfying this protocol — no other code changes are required.

## AimspyMatrix Block-Sparse Format

The `AimspyMatrix` dataclass is AimsPy's canonical in-memory representation:

```python
@dataclass
class AimspyMatrix:
    blocks: dict[tuple[int, int, int, int, int], np.ndarray]  # (R1,R2,R3,i,j) -> (n_i, n_j)
    n_spin: int = 1
```

### Conventions

| Property | Convention |
|----------|------------|
| `R` vector | `R_aimspy = -R_aims` (same sign as DeepH) |
| Atom indices | aims native order (no reordering) |
| Orbital order | aims native basis order (no reordering) |
| Parity | wiki/DeepH convention (`phase_i * phase_j` already applied) |
| Units | Hartree |
| Hermitian partners | **both** `(R,i,j)` and `(-R,j,i)` are stored |

The parity convention is implemented in `AimspyStructure.phase_factor`: `-1` if `m > 0 and m odd`, else `+1`. It is **self-inverse** (`phase² = 1`), so applying it once converts aims ↔ aimspy and applying it again undoes the conversion.

### CSR conversion

`AimspyMatrix` round-trips with FHI-aims' internal CSR layout via:

- `AimspyMatrix.from_aims_csr(h0, csr_descr, structure)` — walks the CSR triplanes, applies `phase_i * phase_j`, sign-flips `R`, and stores both Hermitian partners.
- `AimspyMatrix.to_aims_csr(csr_descr, structure)` — reverse walk, with Hermitian fallback (if `(R,i,j)` is missing, tries `(-R,j,i)` and transposes). Undoes parity. Returns a `(n_spin, n_ham_size)` C-contiguous array ready for `ctypes.memmove` into the Fortran buffer.

> **Note**: both converters are spinless — they read/write channel 0 only. Spin-polarised support is on the roadmap.

## DeepH Data Format

`DeepHData` reads and writes the standard DeepH on-disk format used throughout the DeepH ecosystem. The format is shared with [DeepH-dock](https://github.com/kYangLi/DeepH-dock) — for the full field-level specification, see the [DeepH-dock Key Concepts](https://deeph-dock.readthedocs.io/en/latest/key_concepts.html) page. A summary:

```
some_directory/
├── POSCAR              # Atomic structure (VASP format, element-grouped order)
├── info.json           # System metadata + basis set info
├── overlap.h5          # Overlap matrix S (sparse)
├── hamiltonian.h5      # Hamiltonian H (sparse, eV)
└── hamiltonian_init.h5 # Free-atom initial Hamiltonian (sparse, eV)
```

Each `.h5` file stores four datasets: `atom_pairs` `(N,5)`, `chunk_boundaries` `(N+1,)`, `chunk_shapes` `(N,2)`, and `entries` `(M,)`. The atom order in `POSCAR` is **element-grouped** (different from aims native order); `DeepHData` handles the reordering via `AimspyStructure.atom_permutation`.

### Unit conventions

| Quantity | AimsPy internal | DeepH on-disk |
|----------|-----------------|---------------|
| Hamiltonian | Hartree | eV |
| Overlap | dimensionless | dimensionless |
| Coordinates | Å | Å (in POSCAR) |

`DeepHData.from_memory` converts Hartree → eV on write; `DeepHData.to_aimspy` converts eV → Hartree on read.

## FHI-aims Patch System

The bundled patch (`aimspy/_patches/aimspy-patch_v0.1.0.diff`, ~1100 lines) does three things:

1. **Adds `src/aimspy_api/`** with five Fortran modules:
   - `callback.f90` — `TAimspyCsrMxDescr` (bind(C) struct), `TAimspyCallback` handle type, abstract callback interfaces.
   - `api_bank.f90` — module-level `save` arrays (`c_hamiltonian`, `c_overlap`), `aimspy_energy`, `aimspy_forces` accessors.
   - `info.f90` — `TAimspyInfo` bind(C) struct + `aimspy_get_info` populating a `save` buffer.
   - `register.f90` — the five `aimspy_register_*_callback` bind(C) subroutines.
   - `main.f90` — `aimspy_init` / `aimspy_run` / `aimspy_finalize` / `aimspy_all` lifecycle entry points.

2. **Hooks into `src/initialize_scf.f90`** — the five trigger points after `reshape_matrices`, and the warmstart short-circuit calling `advance_KS_solution` on the injected Hamiltonian with `restart_zero_iteration=.true.`.

3. **Exposes `pbc_lists.f90` arrays** — adds `target` attributes to `index_hamiltonian` / `column_index_hamiltonian` so they can be exposed via `c_loc`.

The patch is **versioned** (currently `v0.1.0`) and managed by the `aimspy patch` CLI, which can apply, uninstall, dry-run, and list bundled versions. Multiple patch versions can ship side-by-side; the CLI auto-detects the currently-applied version by reading a `PATCH_VERSION` line that the patch itself writes into the source tree's `Makefile`.

## Data Flow in AimsPy

A typical warmstart workflow:

1. **Input**: FHI-aims `control.in` + `geometry.in` in `work_dir`; an external Hamiltonian source (e.g. a `DeepHData` directory from a DeepH-trained model).
2. **Processing**:
   - `Calculator.init` loads `libaims.so`, calls `aimspy_init`, builds `AimspyStructure`.
   - `_wire_callbacks` registers the default callbacks based on `CalculatorConfig` flags.
   - `Calculator.calc` calls `aimspy_run`. Inside FHI-aims, after `reshape_matrices`:
     - `get_descr` populates the CSR layout.
     - `export_ovlp` / `export_h0` capture overlap and free-atom `H_init` (if enabled).
     - `python_func` converts the external source via `to_aimspy(structure)`.
     - `modify_h0` applies the `Strategy`, writes the result back via `memmove`, and the patch short-circuits the diagonalisation.
3. **Output**: `calc.hamiltonian` (`AimspyMatrix`, Hartree), `calc.energy` (Hartree), `calc.forces` (eV/Å), and optionally `calc.overlap` / `calc.initial_hamiltonian`. These can be exported to DeepH format via `DeepHData.from_aimspy(...).save(...)`.

For more on the API surface, see [Basic Usage](./basic_usage.md). For extending AimsPy with new callbacks or matrix sources, see the [Development Guide](./for_developers/development_guide.md).
