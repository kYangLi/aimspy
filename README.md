<h1 align="center">aimspy</h1>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/aimspy.svg)](https://pypi.org/project/aimspy/)
[![Python Versions](https://img.shields.io/badge/python-3.12|3.13|3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/pypi/l/aimspy.svg)](https://pypi.org/project/aimspy/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/aimspy.svg)](https://github.com/kYangLi/aimspy/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/aimspy.svg?style=social)](https://github.com/kYangLi/aimspy/stargazers)

*In-memory Python interface to FHI-aims via ctypes, for seamless integration with [DeepX/DeepH-pack](https://github.com/kYangLi/DeepH-pack-docs).*

</div>

**AimSpy** drives [FHI-aims](https://aims-code.rg.mpg.de/) DFT calculations
directly from Python — no subprocess, no file-staged I/O on hot paths — by
loading a patched `libaims.so` via `ctypes` and exchanging matrices in memory
through a callback framework.

> **Status:** `v0.1.0` (alpha) — Calculator lifecycle, ctypes binding,
> callback framework, aimspy standard format, DeepH interface layer, unified
> `modify_init_ham()` API (direct + deferred), forces export, and overlap
> capture are implemented and tested.

- [Features](#features)
- [Installation](#installation)
  - [From PyPI](#from-pypi)
  - [From source (editable, with dev deps)](#from-source-editable-with-dev-deps)
  - [Patching FHI-aims](#patching-fhi-aims)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [DeepH warmstart (1-iteration SCF)](#deeph-warmstart-1-iteration-scf)
  - [Export to DeepH format](#export-to-deeph-format)
  - [Error recovery](#error-recovery)
- [API overview](#api-overview)
  - [`Calculator` — main class](#calculator--main-class)
  - [`CalculatorConfig`](#calculatorconfig)
  - [Other public symbols](#other-public-symbols)
- [Development](#development)
  - [Environment variables](#environment-variables)
- [License](#license)


## Features

- **In-memory SCF** — load `libaims.so` once, drive SCF from Python via `ctypes`
- **MPI-transparent** — works under `mpiexec`; rank-0 vs. all-rank APIs documented
- **Warmstart** — inject an external Hamiltonian (e.g. DeepH prediction) to
  converge SCF in 1 iteration
- **Pluggable matrix sources** — `ExternalMatrixSource` protocol; `DeepHData`
  ships built-in
- **DeepH I/O** — read/write DeepH format (`POSCAR` + `info.json` + `.h5`)
- **Callback framework** — 5 hook points (`get_descr`, `export_ovlp`,
  `export_h0`, `modify_h0`, `python_func`) auto-wrapped to Python
- **Bundled FHI-aims patch** — `aimspy patch` CLI applies/reverses versioned
  diffs; no manual editing of the aims source tree

## Installation

### From PyPI

```bash
pip install aimspy
```

### From source (editable, with dev deps)

```bash
git clone https://github.com/kYangLi/aimspy.git
cd aimspy
pip install -e ".[dev]"
```

Requires Python 3.12–3.14, `numpy>=1.24`, `h5py>=3.0`, `mpi4py>=3.0`, `click>=8.0`.

### Patching FHI-aims

aimspy ships a bundled patch that adapts an FHI-aims source tree to expose
the in-memory interface. Apply it with the `aimspy patch` command:

```bash
cd /path/to/FHI-aims        # clean checkout, e.g. on branch `dev`
aimspy patch                 # applies the latest bundled diff
```

Common variants:

```bash
aimspy patch -v v0.1.0 /path/to/FHI-aims   # specific version
aimspy patch --check /path/to/FHI-aims     # dry-run
aimspy patch --uninstall /path/to/FHI-aims # reverse the detected patch
aimspy patch --list                        # show bundled versions
```

**Prerequisites:** a clean FHI-aims checkout on the patch's base branch
(currently `dev`). The tree must be unpatched; applying on top of an
unrelated branch may fail. By default `git apply` is used on git repos,
falling back to `patch -p1` otherwise (`--no-git` forces `patch(1)`).

Full CLI reference:

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

## Quick start

Baseline SCF on a prepared `work_dir` (containing `control.in` + `geometry.in`):

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig

config = CalculatorConfig(lib_path="/path/to/libaims.so")
with Calculator(config) as calc:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
    H = calc.hamiltonian     # AimspyMatrix (block-sparse, Hartree)
    E = calc.energy          # float (Hartree)
```

Run with MPI:

```bash
mpiexec -np 8 python script.py
```

## Usage

### DeepH warmstart (1-iteration SCF)

Inject a pre-trained DeepH Hamiltonian as the initial guess:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy.interface.deeph import DeepHData

data = DeepHData.from_directory("deeph_warm/")
config = CalculatorConfig(lib_path="/path/to/libaims.so")
calc = Calculator(config)
calc.modify_init_ham(source=data, strategy=Strategy.REPLACE)
calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
```

**Deferred source** — generate the source at runtime (after H0/overlap are
available, inside the `python_func` callback):

```python
config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,
)
calc = Calculator(config)

@calc.modify_init_ham(strategy=Strategy.REPLACE, option={"deeph_path": "deeph_warm/"})
def gen_source(calculator, option):
    # calculator.initial_hamiltonian / .overlap available here
    return DeepHData.from_directory(option["deeph_path"])

calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
```

Other `Strategy` values: `ADD` (add external H to live H0), `SCALE` (scale
H0 by `factor=`), `CUSTOM` (user function via `custom_fn=`).

### Export to DeepH format

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig
from aimspy.interface.deeph import DeepHData

config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,
)
with Calculator(config) as calc:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")

    dd = DeepHData.from_aimspy(
        calc.structure,
        hamiltonian=calc.hamiltonian,
        overlap=calc.overlap,
        initial_hamiltonian=calc.initial_hamiltonian,
    )
    dd.save("deeph_out/")
```

### Error recovery

If SCF crashes, use `force_close()` (always safe, swallows Fortran errors)
and create a fresh `Calculator`:

```python
calc = Calculator(CalculatorConfig(lib_path="..."))
try:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./bad_input")
except Exception:
    calc.force_close()
    # create a new Calculator for the next run
```

## API overview

### `Calculator` — main class

| Method / Property | Description |
|-------------------|-------------|
| `do(comm, work_dir)` | One-shot: `init()` + `calc()`. Common entry point. |
| `init(comm, work_dir)` | Load libaims, call `aimspy_init`, wire callbacks. |
| `calc()` | Run SCF. Raises `AimspyCallbackError` on callback failure. |
| `close()` / `force_close()` | Finalize (graceful / forced from any state). |
| `modify_init_ham(source, *, strategy, factor, custom_fn, option)` | Configure H0 modification (direct or deferred). |
| `register_callback(name, fn, aux, extra_ptr)` | Register custom callback. |
| `info`, `structure` | Runtime snapshot (all ranks). |
| `energy`, `forces` | SCF results (all ranks; Hartree / eV·Å⁻¹). |
| `hamiltonian`, `overlap`, `initial_hamiltonian` | `AimspyMatrix` (rank 0; opt-in flags). |
| `rs_hamiltonian`, `rs_overlap` | Raw CSR flat arrays (rank 0). |
| `csr_descr`, `work_dir`, `comm` | Layout + execution context. |

### `CalculatorConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lib_path` | `Path` | required | Path to patched `libaims.so`. |
| `control_path`, `geometry_path` | `Path` | `None` | Optional inputs to copy into `work_dir`. |
| `initializer` | `callable` | `None` | `fn(Calculator) -> None` on rank 0 before `aimspy_init`. |
| `log_level` | `str` | `"INFO"` | Python logging level. |
| `logfile` | `Path` | `aims.out` | aims log file name. |
| `capture_initial_hamiltonian` | `bool` | `False` | Enable `export_h0` callback. |
| `capture_overlap` | `bool` | `False` | Enable `export_ovlp` callback (all-rank live overlap). |

### Other public symbols

- `Strategy` — enum: `REPLACE`, `ADD`, `SCALE`, `CUSTOM`.
- `CallbackName` — enum: `GET_DESCR`, `EXPORT_OVLP`, `EXPORT_H0`,
  `MODIFY_H0`, `PYTHON_FUNC`.
- `AimspyMatrix` — block-sparse matrix with `blocks` dict,
  `from_aims_csr()` / `to_aims_csr()` converters.
- `AimspyStructure` — structure + orbital descriptor (cached
  `phase_factor`, `basis_subidx`, `atom_permutation`).
- `ExternalMatrixSource` — `Protocol` with `to_aimspy(structure)`.
- `DeepHData` — DeepH format reader/writer; see
  `aimspy.interface.deeph`.

## Development

```bash
make install    # create .venv, install editable with dev deps
make test       # run tests
make lint       # ruff check + black --check
make build      # build wheel
```

### Environment variables

Integration tests and examples require `AIMSPY_TEST_AIMS_LIBPATH` to point
at your patched `libaims.so`:

```bash
export AIMSPY_TEST_AIMS_LIBPATH=/path/to/FHI-aims-deeph/build/libaims.so
```

Optional:

- `AIMSPY_TEST_NPROC` — MPI process count for `tests/test_strategies.py`
  (default: `8`).

## License

AimSpy is released under **GPL-3.0-or-later** (see `LICENSE`).

FHI-aims itself is **not** distributed with AimSpy and remains under its own
licence agreement with the aims team. Users must obtain FHI-aims source code
independently.
