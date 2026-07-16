<h1 align="center">aimspy</h1>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/aimspy.svg)](https://pypi.org/project/aimspy/)
[![Python Versions](https://img.shields.io/badge/python-3.12|3.13|3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/pypi/l/aimspy.svg)](https://pypi.org/project/aimspy/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/aimspy.svg)](https://github.com/kYangLi/aimspy/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/aimspy.svg?style=social)](https://github.com/kYangLi/aimspy/stargazers)

*In-memory Python interface to FHI-aims via ctypes, for seamless integration with [DeepX/DeepH-pack](https://github.com/kYangLi/DeepH-pack-docs).*

</div>

**AimSpy** enables driving [FHI-aims](https://aims-code.rg.mpg.de/)
calculations directly from Python — no subprocess, no file-staged I/O
for hot paths — by linking a small ctypes extension against the
FHI-aims Fortran library.

> **Status:** `v0.1.0` — Calculator, ctypes binding, callback framework,
> aimspy standard format, DeepH interface layer, unified `modify()` API
> (direct + deferred), forces export, and overlap capture are implemented
> and tested. The package is alpha-stage but functional.

## Installation

```bash
pip install aimspy
```

Or install from source in editable mode with development dependencies:

```bash
git clone https://github.com/kYangLi/aimspy.git
cd aimspy
pip install -e ".[dev]"
```

## Patching FHI-aims

aimspy ships a bundled patch that adapts an FHI-aims source tree so it can
expose the in-memory interface aimspy needs.  The patch is applied with the
`aimspy patch` command.

### Prerequisites

- A clean checkout of the FHI-aims source code matching the patch's base
  branch (currently `dev`).  The tree **must be at the unpatched state**;
  applying on top of an unrelated branch may fail or produce conflicts.

### Apply the patch

From inside the FHI-aims source directory:

```bash
cd /path/to/FHI-aims        # clean checkout, e.g. on branch `dev`
aimspy patch
```

Or point at it from anywhere:

```bash
aimspy patch /path/to/FHI-aims
```

A specific bundled version can be requested:

```bash
aimspy patch -v v0.1.0 /path/to/FHI-aims
```

- If no patch is detected, the latest bundled version is applied directly.
- If a patch is already detected (the source tree's root `Makefile` records
  `PATCH_VERSION`), you are asked whether to uninstall it first and then
  apply the new one.  Pass `-y` to skip the confirmation.

### Uninstall the patch

```bash
aimspy patch --uninstall /path/to/FHI-aims
```

This reverses the **currently detected** version using its matching bundled
diff, restoring the source tree to its unpatched state.

### Dry-run / list

```bash
aimspy patch --check /path/to/FHI-aims   # validate without modifying files
aimspy patch --list                       # show bundled patch versions
```

### Non-git source trees

By default `git apply` is used when the target is a git repository, falling
back to `patch -p1` otherwise.  `--no-git` forces `patch(1)`:

```bash
aimspy patch --no-git /path/to/FHI-aims
```

### Reference

```
aimspy patch [SOURCE] [OPTIONS]

Arguments:
  SOURCE                 FHI-aims source directory (default: current dir)

Options:
  -v, --version TEXT     Patch version to apply (default: latest)
  -l, --list             List bundled patches and exit
  --check, --dry-run     Dry-run only; do not modify the tree
  --uninstall            Reverse the currently-detected patch
  --no-git               Force patch(1) instead of git apply
  -y, --yes              Skip confirmation prompts
```

## API Reference

### `Calculator` — main class

| Method / Property | Description |
|-------------------|-------------|
| `do(comm, work_dir)` | One-shot: `init()` + `calc()`. Common entry point. |
| `init(comm, work_dir)` | Load libaims, call `aimspy_init`, wire callbacks. |
| `calc()` | Run SCF. Raises `AimspyCallbackError` on callback failure. |
| `close()` | Finalize. No-op in UNINIT/FINALIZED; raises in RUNNING. |
| `force_close()` | Force-finalize from any state (swallows Fortran errors). |
| `modify(source, *, strategy, factor, custom_fn, aux)` | Configure H0 modification (direct or deferred via decorator). |
| `register_callback(name, fn, aux, extra_ptr)` | Register custom callback (`name`: `CallbackName` or `str`). |
| `callback_registered(name)` | Check if a callback is registered. |
| `info` | `AimspyInfo` — runtime snapshot (all ranks). |
| `structure` | `AimspyStructure` — structure + orbital info (all ranks). |
| `energy` | `float` — SCF total energy in Hartree (all ranks). |
| `forces` | `Optional[np.ndarray]` `(n_atoms, 3)` — eV/Å (all ranks). |
| `hamiltonian` | `AimspyMatrix` — converged H (rank 0). |
| `overlap` | `AimspyMatrix` — live (if `capture_overlap`) or fallback (rank 0). |
| `initial_hamiltonian` | `Optional[AimspyMatrix]` — free-atom H_init (if `capture_initial_hamiltonian`). |
| `csr_descr` | `Optional[CsrMatrixDescriptor]` — CSR layout. |
| `rs_hamiltonian` | `np.ndarray` — raw CSR flat (rank 0). |
| `rs_overlap` | `np.ndarray` — raw overlap flat (rank 0). |
| `work_dir`, `comm` | Execution context. |

### `CalculatorConfig` — configuration dataclass

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lib_path` | `Path` | required | Path to patched `libaims.so`. |
| `control_path` | `Path` | `None` | Optional `control.in` to copy into `work_dir`. |
| `geometry_path` | `Path` | `None` | Optional `geometry.in` to copy into `work_dir`. |
| `initializer` | `callable` | `None` | `fn(Calculator) -> None` on rank 0 before `aimspy_init`. |
| `log_level` | `str` | `"INFO"` | Python logging level. |
| `logfile` | `Path` | `aims.out` | aims log file name. |
| `capture_initial_hamiltonian` | `bool` | `False` | Enable `export_h0` callback. |
| `capture_overlap` | `bool` | `False` | Enable `export_ovlp` callback (all-rank live overlap). |

### `Strategy` — H0 modification enum

| Value | Description |
|-------|-------------|
| `REPLACE` | Replace live H_init with external source. |
| `ADD` | Add external source to live H_init. |
| `SCALE` | Scale live H_init by `factor`. |
| `CUSTOM` | Custom function `fn(live, external, structure, aux)`. |

### `CallbackName` — callback type enum

`GET_DESCR`, `EXPORT_OVLP`, `EXPORT_H0`, `MODIFY_H0`, `PYTHON_FUNC`.

### `AimspyMatrix` — block-sparse matrix

| Member | Description |
|--------|-------------|
| `blocks` | `dict[(R1,R2,R3,i_atom,j_atom), np.ndarray]` — Hartree, aims order. |
| `n_spin` | `int` |
| `n_pairs` | `int` — number of blocks. |
| `from_aims_csr(h0, csr_descr, structure)` | Convert aims CSR flat → aimspy blocks. |
| `to_aims_csr(csr_descr, structure)` | Convert aimspy blocks → aims CSR flat. |

### `AimspyStructure` — structure + orbital descriptor

| Member | Description |
|--------|-------------|
| `n_atoms`, `n_basis`, `n_spin`, `n_periodic` | Scalar dimensions. |
| `lattice`, `atom_symbols`, `atom_coords` | Structure data. |
| `basis_atom`, `basis_l`, `basis_m` | Per-basis-function orbital info. |
| `phase_factor` | Wiki/DeepH parity (cached). |
| `basis_subidx` | Per-atom orbital sub-index (cached). |
| `orbit_per_atom` | Basis function count per atom (cached). |
| `atom_permutation` | `(old2new, new2old)` aims↔POSCAR mapping (cached). |

### `ExternalMatrixSource` — Protocol

Any object with `to_aimspy(structure) -> AimspyMatrix` satisfies this protocol.
`DeepHData` is the built-in implementation.

### `DeepHData` — DeepH format reader/writer

| Method | Description |
|--------|-------------|
| `from_directory(path)` | Read POSCAR + info.json + `.h5` files. |
| `from_memory(...)` | Build from in-memory block dicts. |
| `from_aimspy(structure, hamiltonian, overlap, initial_hamiltonian, template)` | Convert from aimspy format. |
| `to_aimspy(structure)` | Convert to aimspy `AimspyMatrix` (satisfies `ExternalMatrixSource`). |
| `set_hamiltonian(matrix, structure)` | Set H from `AimspyMatrix`. |
| `set_overlap(matrix, structure)` | Set S from `AimspyMatrix`. |
| `set_initial_hamiltonian(matrix, structure)` | Set H_init from `AimspyMatrix`. |
| `save(path)` | Write all non-None content. |
| `save_metadata(path)` | Write POSCAR + info.json only. |
| `save_hamiltonian(path)` / `save_overlap(path)` / `save_initial_hamiltonian(path)` | Write individual `.h5` files. |

## Usage

### Baseline SCF

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig

config = CalculatorConfig(lib_path="/path/to/libaims.so")
with Calculator(config) as calc:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
    H = calc.hamiltonian     # AimspyMatrix (available by default)
    E = calc.energy          # float (Hartree)
```

### DeepH warmstart

**Direct source** — pre-built `DeepHData`:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy.interface.deeph import DeepHData

data = DeepHData.from_directory("deeph_warm/")
config = CalculatorConfig(lib_path="/path/to/libaims.so")
calc = Calculator(config)
calc.modify(source=data, strategy=Strategy.REPLACE)
calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
# SCF converges in 1 iteration
```

**Deferred source** — source generated at runtime during the
`python_func` callback (after H0/overlap are available):

```python
config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,  # enables calc.initial_hamiltonian
)
calc = Calculator(config)

@calc.modify(strategy=Strategy.REPLACE, aux={"deeph_path": "deeph_warm/"})
def gen_source(calculator, aux):
    # calculator.initial_hamiltonian / .overlap available here
    return DeepHData.from_directory(aux["deeph_path"])

calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
```

Other strategies: `Strategy.ADD` (add external source to live H0),
`Strategy.SCALE` (scale live H0 by `factor=`), `Strategy.CUSTOM`
(custom function via `custom_fn=`).

### Export to DeepH format

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig
from aimspy.interface.deeph import DeepHData

config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,   # opt in to free-atom H_init capture
)
with Calculator(config) as calc:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
    hamiltonian = calc.hamiltonian
    overlap = calc.overlap
    initial_hamiltonian = calc.initial_hamiltonian  # available because
                                                    # capture_initial_hamiltonian=True

    dd = DeepHData.from_aimspy(
        calc.structure,
        hamiltonian=hamiltonian,
        overlap=overlap,
        initial_hamiltonian=initial_hamiltonian,
    )
    dd.save("deeph_out/")
```

### Advanced usage

**Error recovery** — if SCF crashes, use `force_close()` then create a new
Calculator:

```python
calc = Calculator(CalculatorConfig(lib_path="..."))
try:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./bad_input")
except Exception:
    calc.force_close()  # always works, swallows Fortran errors
    # create a new Calculator for the next run
```

**Custom strategy** — modify H0 with a user function:

```python
import numpy as np

def zero_offsite(live, external, structure, aux):
    """Zero out all non-self-pair blocks."""
    for key in list(live.blocks.keys()):
        R1, R2, R3, i, j = key
        if (R1, R2, R3) != (0, 0, 0) or i != j:
            live.blocks[key] = np.zeros_like(live.blocks[key])

calc.modify(strategy=Strategy.CUSTOM, custom_fn=zero_offsite)
```

**Register callback** (enum or string, before or after init):

```python
from aimspy import CallbackName

calc.register_callback(CallbackName.EXPORT_H0, my_export_fn, aux={})
# or: calc.register_callback("export_h0", my_export_fn, aux={})
```

**Logging**: INFO/WARNING on rank 0 only; ERROR on all ranks.

## Development

```bash
make install    # create .venv, install editable with dev deps
make test       # run tests
make lint       # ruff check + black --check
make build      # build wheel
```

## Licensing

AimSpy is released under **GPL-3.0-or-later** (see `LICENSE`).

FHI-aims itself is **not** distributed with AimSpy and remains under its
own licence agreement with the aims team. Users must obtain FHI-aims
source code independently.
