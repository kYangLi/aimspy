<!-- markdownlint-disable MD033 MD036 -->
<h1><p align="center">
  <img src="https://raw.githubusercontent.com/kYangLi/aimspy/main/docs/_image/logo-large.svg" alt="AimsPy Logo" width="500">
</p></h1>

<div align="center">

### In-memory Python interface to FHI-aims

[![GitHub Actions PyPI Release](https://github.com/kYangLi/aimspy/actions/workflows/publish.yaml/badge.svg)](https://github.com/kYangLi/aimspy/actions/workflows/publish.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/aimspy.svg)](https://pypi.org/project/aimspy/)
[![Python 3.12–3.14](https://img.shields.io/badge/python-3.12–3.14-blue.svg)](https://www.python.org/)

[![License](https://img.shields.io/pypi/l/aimspy.svg)](https://pypi.org/project/aimspy/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/aimspy.svg)](https://github.com/kYangLi/aimspy/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/aimspy.svg?style=social)](https://github.com/kYangLi/aimspy/stargazers)

*For seamless integration with [DeepX/DeepH-pack](https://github.com/kYangLi/DeepH-pack-docs)*
</div>

*AimsPy* drives [FHI-aims](https://www.fhi-aims.org/) DFT calculations directly from Python (no subprocess, no file-staged I/O on hot paths) by loading a patched `libaims.so` via `ctypes` and exchanging matrices in memory through a callback framework. 

It is designed as the FHI-aims binding layer of the [DeepH](https://github.com/kYangLi/DeepH-pack-docs) ecosystem, and the central enabler of **warmstart SCF**: injecting an externally-predicted Hamiltonian (e.g. from a DeepH-trained model) as the initial guess so that SCF converges rapidly in several iterations.

For the most comprehensive usage documentation, please visit [https://docs.deeph-pack.com/aimspy/en/latest/](https://docs.deeph-pack.com/aimspy/en/latest/).

---

- [Core Features](#core-features)
- [Runtime Environment](#runtime-environment)
  - [Install Intel OneAPI Toolchain](#install-intel-oneapi-toolchain)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Citation](#citation)
- [Application Scenarios](#application-scenarios)
- [Contributing](#contributing)
- [License](#license)
- [Support \& Contact](#support--contact)

## Core Features

- **Bundled FHI-aims Patch:** 
    Patch FHI-aims with a single command `aimspy patch` applies, uninstalls, and lists versioned patches against an FHI-aims source tree. No manual code editing required.

- **In-Memory SCF:** 
  Run FHI-aims SCF calculations directly from Python. Hamiltonian, overlap, energy, and forces are returned as native Python objects, ready for analysis or downstream processing.

- **DeepH Export:** 
  Export converged Hamiltonian, overlap, and free-atom initial Hamiltonian to the DeepH on-disk format in a single pipeline, ideal for generating training data for DeepH models.

- **Warmstart:** 
  Provide a pre-trained Hamiltonian (e.g. from a DeepH model) as the initial guess, and SCF converges in several iterations instead of the usual 10+. Strategies (`REPLACE`, `ADD`, `SCALE`, `CUSTOM`) cover warmstart, correction (Delta-prediction), scaling, and custom transforms.

- **Pluggable Matrix Sources:** 
  Use any Hamiltonian source for warmstart. The built-in `DeepHData` adapter reads DeepH-format data directly, and adding a new format is just one subpackage under `aimspy/interface/`.


## Runtime Environment

AimsPy requires a patched `libaims.so` built with an MPI-enabled Fortran compiler and a BLAS/LAPACK math library. The tested configuration uses Intel OneAPI.

### Install Intel OneAPI Toolchain

Download from the [Intel OneAPI Toolkit page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit-download.html).

Required components:
- Intel Fortran compiler (`ifx` / `mpiifx`)
- Intel C/C++ compiler (`icx` / `icpx`)
- Intel MKL (includes BLAS, LAPACK, ScaLAPACK, BLACS)
- Intel MPI (`mpiifx` is the MPI Fortran wrapper)

After installation, set up the environment:

```bash
source /opt/intel/oneapi/setvars.sh
```

> **Note**: Other MPI distributions and math libraries (e.g. OpenMPI + OpenBLAS) can also be used, as long as they support building FHI-aims and `mpi4py`. The key requirement is that `mpi4py` and `libaims.so` use the **same MPI backend** — see [Installation & Setup](https://docs.deeph-pack.com/aimspy/en/latest/installation_and_setup.html) for details.

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

**Prerequisites:**
  a clean FHI-aims checkout on the patch's base branch. FHI-aims itself is **not** distributed with AimsPy. Users must obtain its source code independently from the [aims team](https://fhi-aims.org/get-the-code-menu/get-the-code).

> **Note**: The current patch supports FHI-aims versions **250822** and **250822_1** only. Other versions are not compatible. Patches for additional FHI-aims versions will be released in the future.

> **Note**: AimsPy loads `libaims.so` via `ctypes` at runtime, so FHI-aims must be built as a **shared library** (`-DBUILD_SHARED_LIBS=ON`). For detailed setup (uv environment, building `libaims.so`, environment variables), see [Installation & Setup](https://docs.deeph-pack.com/aimspy/en/latest/installation_and_setup.html) for full build instructions.

### Basic Usage

Three core workflows:

1. **Baseline SCF**:
   run a standard FHI-aims SCF calculation and extract results as Python objects.
2. **DeepH export**:
   run SCF and export matrices to the DeepH on-disk format for training data generation.
3. **DeepH warmstart**:
   inject a pre-trained Hamiltonian and converge SCF in several iterations.

**Baseline SCF**
on a prepared `work_dir` (**must contain** the FHI-aims required input files `control.in` + `geometry.in`):

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig

comm = MPI.COMM_WORLD
rank = comm.rank

config = CalculatorConfig(lib_path="/path/to/libaims.so")
with Calculator(config) as calc:
    calc.do(comm=comm, work_dir="./MoS2")
    if rank == 0:
      H = calc.hamiltonian     # AimspyMatrix (block-sparse, Hartree, rank-0 only)
    E = calc.energy          # float (Hartree)
```

Run with MPI:

```bash
mpiexec -np 8 python script.py
```

> **Note**: Matrix extraction and injection (e.g. warmstart, overlap/H0
> capture) require a **periodic system** with `use_local_index = .false.`
> in `control.in`. Forward SCF works with any system type. For isolated
> molecules, use a large periodic cell with vacuum. See the
> [examples](https://github.com/kYangLi/aimspy/tree/main/examples).

**DeepH export**
export converged matrices to DeepH format:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig
from aimspy import DeepHData

comm = MPI.COMM_WORLD
rank = comm.rank

config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,  # capture free-atom H0
)
with Calculator(config) as calc:
    calc.do(comm=comm, work_dir="./MoS2")

    # Export H, S, H0 to DeepH on-disk format
    if rank == 0:
      dd = DeepHData.from_aimspy(
          calc.structure,
          hamiltonian=calc.hamiltonian,
          overlap=calc.overlap,
          initial_hamiltonian=calc.initial_hamiltonian,
      )
      dd.save("deeph_out/")
```

> **Note**: For the DeepH on-disk data format specification (POSCAR, info.json,
> .h5 files), see [DeepH-dock Key Concepts](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html).

**DeepH warmstart**
inject a pre-trained Hamiltonian as the initial guess:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy import DeepHData

data = DeepHData.from_directory("deeph_out/")
config = CalculatorConfig(lib_path="/path/to/libaims.so")
calc = Calculator(config)
calc.modify_init_ham(source=data, strategy=Strategy.REPLACE)
calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
```

For more information on deferred source, overlap capture, error recovery, and the full API, see [Basic Usage](https://docs.deeph-pack.com/aimspy/en/latest/basic_usage.html) and [API Reference](https://docs.deeph-pack.com/aimspy/en/latest/api_reference.html).

## Citation

Since AimsPy is part of the DeepH ecosystem and drives FHI-aims calculations, we recommend citing the following papers:

**1. DeepH-pack** — the complete package featuring the latest implementation, methodology, and workflow of [DeepH](https://github.com/kYangLi/DeepH-pack-docs):

[Yang Li, Yanzhen Wang, Boheng Zhao, *et al*. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)](https://arxiv.org/abs/2601.02938)

```bibtex
@article{li2026deeph,
    title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
    author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
    journal={arXiv preprint arXiv:2601.02938},
    year={2026}
}
```

**2. DeepH-aims** — the paper describing the DeepH–FHI-aims integration workflow (in publishing):

<!-- TODO: fill in the DeepH-aims paper citation once published. -->
[Authors]. [Title]. [Journal], in publishing.

**3. FHI-aims** — the original FHI-aims paper, since AimsPy drives FHI-aims calculations:

[Volker Blum, Ralf Gehrke, Felix Hanke, Paula Havu, Ville Havu, Xinguo Ren, Karsten Reuter, Matthias Scheffler. Ab initio molecular simulations with numeric atom-centered orbitals. Computer Physics Communications 180(11), 2175–2196 (2009)](https://doi.org/10.1016/j.cpc.2009.06.022)

```bibtex
@article{BLUM20092175,
    title = {Ab initio molecular simulations with numeric atom-centered orbitals},
    journal = {Computer Physics Communications},
    volume = {180},
    number = {11},
    pages = {2175--2196},
    year = {2009},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2009.06.022},
    url = {https://www.sciencedirect.com/science/article/pii/S0010465509002033},
    author = {Volker Blum and Ralf Gehrke and Felix Hanke and Paula Havu and Ville Havu and Xinguo Ren and Karsten Reuter and Matthias Scheffler},
    keywords = {molecular simulations, Density-functional theory, Atom-centered basis functions, Hartree--Fock, MP2, O(N) DFT, self-energy}
}
```

## Application Scenarios

- **DeepH Training Data Generation:**
  Run baseline SCF and export to the DeepH on-disk format (`POSCAR` + `info.json` + `.h5`) in a single pipeline.
- **DeepH Warmstart:** 
  Inject a pre-trained DeepH Hamiltonian as the initial guess and converge SCF in several iterations, enabling rapid downstream property evaluation.
- **FHI-aims Post-Processing:**
  Extract converged Hamiltonian, overlap, and free-atom `H_init` matrices in the standard `AimspyMatrix` format for analysis or conversion.
- **Method Development:**
  Prototype new initial-guess strategies via the `Strategy.CUSTOM` hook, or plug in alternative DFT backends by implementing the `ExternalMatrixSource` protocol.

## Contributing

We welcome contributions from the community! AimsPy is built with a layered architecture (public API → callback framework → ctypes binding → FHI-aims patch), and extension points are deliberately narrow and well-documented.

Common contribution targets:

- **New external matrix sources**
  implement the `ExternalMatrixSource` protocol in a new subpackage under `aimspy/interface/<your_format>/`.
- **New callback hook points**
  follow the extension contract documented in the [Development Guide](https://docs.deeph-pack.com/aimspy/en/latest/for_developers/development_guide.html).
- **New modification strategies**
  extend the `Strategy` enum and the `_apply_strategy` dispatcher.

For the complete development workflow, code style, testing requirements, and pull request process, see the [Development Guide](https://docs.deeph-pack.com/aimspy/en/latest/for_developers/development_guide.html) and [Collaboration Guide](https://docs.deeph-pack.com/aimspy/en/latest/for_developers/collaboration_guide.html).

```bash
make install    # create .venv, install editable with dev deps
make test       # run unit tests (pytest -v)
make lint       # ruff check + black --check
make build      # build sdist + wheel
```

## License

This project is licensed under **GPL-3.0-or-later**. See the [LICENSE](LICENSE) file for details.

FHI-aims itself is **not** distributed with AimsPy and remains under its own licence agreement with the aims team. Users must obtain FHI-aims source code independently.

## Support & Contact

- 📖 **Documentation**: [https://docs.deeph-pack.com/aimspy/en/latest/](https://docs.deeph-pack.com/aimspy/en/latest/)
- 🐛 **Issue Reporting**: [GitHub Issues](https://github.com/kYangLi/aimspy/issues)

---

*AimsPy is the FHI-aims binding layer of the DeepH ecosystem, aiming to promote openness and reproducibility in computational materials science research.*
