# aimspy

**AimSpy** is an in-memory Python interface to [FHI-aims](https://aims-code.rg.mpg.de/)
via pybind11, designed for tight, memory-level integration with
[DeepH/DeepX](https://github.com/deepmodeling/DeepH-Pack).

It enables driving FHI-aims calculations directly from Python — no
subprocess, no file-staged I/O for hot paths — by linking a small
pybind11 extension against the FHI-aims Fortran library.

> **Status:** `v0.0.1` is a minimal placeholder release for reserving
> the package name on PyPI. The frontend (Python object model for aims
> inputs), backend (pybind11 binding), and test suite (DFT benchmark
> cases) are under active development and will follow in subsequent
> versions.

## Licensing

AimSpy is released under **GPL-3.0-or-later** (see `LICENSE`).

FHI-aims itself is **not** distributed with AimSpy and remains under its
own licence agreement with the aims team. Users must obtain FHI-aims
source code independently.
