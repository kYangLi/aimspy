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
> aimspy standard format, and DeepH interface layer are implemented
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

## Usage

### Baseline SCF

```python
from aimspy import Calculator, CalculatorConfig

with Calculator(CalculatorConfig(
    lib_path="/path/to/libaims.so",
    work_dir="./MoS2",
)) as calc:
    calc.run()
    H = calc.hamiltonian     # AimspyMatrix
    E = calc.energy          # float (Hartree)
```

### DeepH warmstart

```python
from aimspy.interface.deeph import DeepHData, DeepHSource

deeph_data = DeepHData.from_directory("deeph_warm/")
calc.modify_h0(source=DeepHSource(deeph_data))
calc.init(comm)
calc.run()
# SCF converges in 1 iteration
```

### Export to DeepH format

```python
calc.run()
H_aimspy = calc.hamiltonian
S_aimspy = calc.overlap
H0 = calc.initial_hamiltonian  # requires calc.capture_h0 = True

from aimspy.interface.deeph import DeepHData
dd = DeepHData.from_aimspy(calc.structure, H=H_aimspy, S=S_aimspy, H0=H0)
dd.save("deeph_out/")
```

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
