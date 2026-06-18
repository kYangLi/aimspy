<h1 align="center">aimspy</h1>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/aimspy.svg)](https://pypi.org/project/aimspy/)
[![Python Versions](https://img.shields.io/badge/python-3.12|3.13|3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/pypi/l/aimspy.svg)](https://pypi.org/project/aimspy/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/aimspy.svg)](https://github.com/kYangLi/aimspy/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/aimspy.svg?style=social)](https://github.com/kYangLi/aimspy/stargazers)

*In-memory Python interface to FHI-aims via pybind11, for seamless integration with [DeepX/DeepH-pack](https://github.com/kYangLi/DeepH-pack-docs).*

</div>

**AimSpy** enables driving [FHI-aims](https://aims-code.rg.mpg.de/)
calculations directly from Python — no subprocess, no file-staged I/O
for hot paths — by linking a small pybind11 extension against the
FHI-aims Fortran library.

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
