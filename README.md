<!-- markdownlint-disable MD033 MD036 -->
<h1><p align="center">
  <img src="https://raw.githubusercontent.com/kYangLi/aimspy/main/docs/_image/logo-large.svg" alt="AimsPy Logo" width="500">
</p></h1>

<div align="center">

### In-memory Python interface to FHI-aims via ctypes

[![GitHub Actions PyPI Release](https://github.com/kYangLi/aimspy/actions/workflows/publish.yaml/badge.svg)](https://github.com/kYangLi/aimspy/actions/workflows/publish.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/aimspy.svg)](https://pypi.org/project/aimspy/)
[![Python 3.12–3.14](https://img.shields.io/badge/python-3.12–3.14-blue.svg)](https://www.python.org/)

[![License](https://img.shields.io/pypi/l/aimspy.svg)](https://pypi.org/project/aimspy/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/aimspy.svg)](https://github.com/kYangLi/aimspy/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/aimspy.svg?style=social)](https://github.com/kYangLi/aimspy/stargazers)

*For seamless integration with [DeepX/DeepH-pack](https://github.com/kYangLi/DeepH-pack-docs)*
</div>

*AimsPy* drives [FHI-aims](https://aims-code.rg.mpg.de/) DFT calculations directly from Python — no subprocess, no file-staged I/O on hot paths — by loading a patched `libaims.so` via `ctypes` and exchanging matrices in memory through a callback framework. It is designed as the FHI-aims binding layer of the [DeepH](https://github.com/kYangLi/DeepH-pack-docs) ecosystem, and the central enabler of **warmstart SCF**: injecting an externally-predicted Hamiltonian (e.g. from a DeepH-trained model) as the initial guess so that a single SCF iteration reproduces the converged result.

At the core of *AimsPy* is **a unified in-memory representation of block-sparse real-space matrices** — `AimspyMatrix` — that round-trips between FHI-aims' internal CSR layout and the DeepH on-disk format with documented sign/parity conventions, making it equally useful as a standalone post-processing interface for FHI-aims users.

For the most comprehensive usage documentation, please visit [https://aimspy.readthedocs.io](https://aimspy.readthedocs.io).

---

- [Core Features](#core-features)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Citation](#citation)
- [Application Scenarios](#application-scenarios)
- [Contributing](#contributing)
- [License](#license)
- [Support & Contact](#support--contact)

## Core Features

- **In-Memory SCF:** Load `libaims.so` once and drive the full SCF cycle from Python via `ctypes`. No subprocess, no file-staged I/O on the hot path — Hamiltonian, overlap, energy, and forces are exchanged as in-memory arrays through a callback framework.

- **Warmstart:** Inject an external Hamiltonian (e.g. a DeepH prediction) as the initial guess and converge SCF in a single iteration. Four strategies — `REPLACE`, `ADD`, `SCALE`, `CUSTOM` — cover warmstart, perturbation, scaling, and arbitrary user transforms.

- **Pluggable Matrix Sources:** The `ExternalMatrixSource` protocol accepts any object with `to_aimspy(structure) -> AimspyMatrix`. A reference `DeepHData` adapter ships built-in; adding a new format is a single subpackage under `aimspy/interface/`.

- **MPI-Transparent:** Works under `mpiexec`; rank-0 vs. all-rank APIs are documented per property. INFO/WARNING messages are emitted on rank 0 only; ERROR on all ranks for debugging.

- **Bundled FHI-aims Patch:** `aimspy patch` applies, uninstalls, and lists versioned diffs against an FHI-aims source tree — no manual editing. The patch exposes five Fortran callback hook points and the warmstart short-circuit inside `initialize_scf.f90`.

## Quick Start

### Installation

Publish version:

```bash
pip install aimspy
```

Development version:

```bash
pip install git+https://github.com/kYangLi/aimspy
```

AimsPy loads a *patched* `libaims.so` at runtime. To patch an FHI-aims source tree:

```bash
cd /path/to/FHI-aims        # clean checkout, e.g. on branch `dev`
aimspy patch                 # applies the latest bundled diff
```

Common variants:

```bash
aimspy patch /path/to/FHI-aims             # patch a specific tree
aimspy patch -v v0.1.0 /path/to/FHI-aims   # use a specific patch version
aimspy patch --check /path/to/FHI-aims     # dry-run
aimspy patch --uninstall /path/to/FHI-aims # reverse the detected patch
aimspy patch --list                        # show bundled versions
```

**Prerequisites:** a clean FHI-aims checkout on the patch's base branch (currently `dev`). FHI-aims itself is **not** distributed with AimsPy — users must obtain its source code independently from the [aims team](https://aims-code.rg.mpg.de/).

For detailed setup (uv environment, building `libaims.so`, environment variables), see [Installation & Setup](https://aimspy.readthedocs.io/en/latest/installation_and_setup.html).

### Basic Usage

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

DeepH warmstart (1-iteration SCF):

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

For complete examples — warmstart with deferred source, DeepH-format export, overlap capture, and error recovery — see [Basic Usage](https://aimspy.readthedocs.io/en/latest/basic_usage.html).

## Citation

*Any and all use of this software, in whole or in part, should clearly acknowledge and link to this repository.*

If you use this code in your academic work, please cite **the complete package featuring the latest implementation, methodology, and workflow of [DeepH](https://github.com/kYangLi/DeepH-pack-docs)**:

[Yang Li, Yanzhen Wang, Boheng Zhao, *et al*. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)](https://arxiv.org/abs/2601.02938)

```bibtex
@article{li2026deeph,
    title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
    author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
    journal={arXiv preprint arXiv:2601.02938},
    year={2026}
}
```

## Application Scenarios

- **DeepH Warmstart:** Inject a pre-trained DeepH Hamiltonian as the initial guess and converge SCF in a single iteration, enabling rapid downstream property evaluation.
- **FHI-aims Post-Processing:** Extract converged Hamiltonian, overlap, and free-atom `H_init` matrices in the standard `AimspyMatrix` format for analysis or conversion.
- **DeepH Training Data Generation:** Run baseline SCF and export to the DeepH on-disk format (`POSCAR` + `info.json` + `.h5`) in a single pipeline.
- **Method Development:** Prototype new initial-guess strategies via the `Strategy.CUSTOM` hook, or plug in alternative DFT backends by implementing the `ExternalMatrixSource` protocol.

## Contributing

We welcome contributions from the community! AimsPy is built with a layered architecture (public API → callback framework → ctypes binding → FHI-aims patch), and extension points are deliberately narrow and well-documented.

Common contribution targets:

- **New external matrix sources** — implement the `ExternalMatrixSource` protocol in a new subpackage under `aimspy/interface/<your_format>/`.
- **New callback hook points** — follow the four-place extension contract (Fortran patch + `callback_types.py` + `prototypes.py` + `registry.py`).
- **New modification strategies** — extend the `Strategy` enum and the `_apply_strategy` dispatcher.

For the complete development workflow, code style, testing requirements, and pull request process, see the [Development Guide](https://aimspy.readthedocs.io/en/latest/for_developers/development_guide.html) and [Collaboration Guide](https://aimspy.readthedocs.io/en/latest/for_developers/collaboration_guide.html).

```bash
make install    # create .venv, install editable with dev deps
make test       # run unit tests (pytest -v)
make lint       # ruff check + black --check
make build      # build sdist + wheel
```

## License

This project is licensed under **GPL-3.0-or-later** — see the [LICENSE](LICENSE) file for details.

FHI-aims itself is **not** distributed with AimsPy and remains under its own licence agreement with the aims team. Users must obtain FHI-aims source code independently.

## Support & Contact

- 📖 **Documentation**: [https://aimspy.readthedocs.io](https://aimspy.readthedocs.io)
- 🐛 **Issue Reporting**: [GitHub Issues](https://github.com/kYangLi/aimspy/issues)

---

*AimsPy is the FHI-aims binding layer of the DeepH ecosystem, aiming to promote openness and reproducibility in computational materials science research.*
