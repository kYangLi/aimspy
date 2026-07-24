# Installation & Setup

AimsPy is a pure-Python package, but it loads a **patched** `libaims.so` at runtime. This page covers installing the Python package, patching an FHI-aims source tree, building the patched library, and configuring the environment variables used by tests and examples.

## Install UV

To begin, configure your environment with `uv`, a fast and versatile Python package manager written in Rust. Please follow the installation instructions on the [official uv website](https://docs.astral.sh/uv/#installation).

On Linux or macOS, you can install `uv` with a single command (requires an internet connection):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It is highly recommended that configuring high-performance mirrors based on your IP location. For example, for users in China, you could use the mirror provided by [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/):

```bash
# Add the following lines into ~/.config/uv/uv.toml
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
```

## Create the AimsPy Python virtual environment

Create a `python 3.13` environment with `uv`:

```bash
mkdir ~/.uvenv
cd ~/.uvenv
uv venv aimspy --python=3.13   # Create the `aimspy` venv in the current dir
```

The uv virtual environment can then be activated with:

```bash
source ~/.uvenv/aimspy/bin/activate
```

All packages installed into the `aimspy` venv will be located under `~/.uvenv/aimspy/`.

AimsPy requires Python 3.12–3.14, `numpy>=1.24`, `h5py>=3.0`, `mpi4py>=3.0`, and `click>=8.0`.

## Prerequisites: Intel OneAPI Environment

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

> **Warning**
>
> `mpi4py` (used by AimsPy) and `libaims.so` **must use the same MPI
> backend**. Sourcing the Intel OneAPI environment before installing
> AimsPy and before building `libaims.so` ensures `mpi4py` compiles
> against the same MPI library that `libaims.so` links against.
>
> Other MPI distributions and math libraries (e.g. OpenMPI + OpenBLAS)
> can also be used, as long as they support building FHI-aims and
> `mpi4py` with a consistent MPI backend.

## Quick Install (for Common Users)

Ensure you've activated the uv environment as described above, then install AimsPy from PyPI:

```bash
uv pip install aimspy
```

For the development version:

```bash
pip install git+https://github.com/kYangLi/aimspy
```

> **Note**: During installation, an internet connection is required. AimsPy and DeepH-pack can be installed under the same Python venv.

## Patching FHI-aims

AimsPy exchanges matrices with FHI-aims through a callback framework exposed by a small Fortran patch. The patch adds a new `src/aimspy_api/` directory to FHI-aims and inserts trigger points inside `src/initialize_scf.f90`, plus the warmstart short-circuit that calls `advance_KS_solution` directly on an injected Hamiltonian.

Apply the latest bundled patch to a clean FHI-aims checkout:

```bash
cd /path/to/FHI-aims        # clean checkout
aimspy patch                 # applies the latest bundled diff
```

Common variants:

```bash
aimspy patch -v v0.1.0 /path/to/FHI-aims   # specific version
aimspy patch --check /path/to/FHI-aims                   # dry-run
aimspy patch --uninstall /path/to/FHI-aims               # reverse the detected patch
aimspy patch --list                                      # show bundled versions
```

**Prerequisites:** a clean FHI-aims checkout on the patch's base branch. The tree must be unpatched; applying on top of an unrelated branch may fail. By default `git apply` is used on git repos, falling back to `patch -p1` otherwise (`--no-git` forces `patch(1)`).

> **Note**: The current patch supports FHI-aims versions **250822** and **250822_1** only. Other versions are not compatible. Patches for additional FHI-aims versions will be released in the future.

Full CLI reference:

```
aimspy patch [SOURCE] [OPTIONS]

Arguments:
  SOURCE                 FHI-aims source directory (default: current dir)

Options:
  -v, --patch-version TEXT  Patch version to apply (default: latest)
  -l, --list             List bundled patches and exit
  --check, --dry-run     Dry-run only; do not modify the tree
  --uninstall            Reverse the currently-detected patch
  --no-git               Force patch(1) instead of git apply
  -y, --yes              Skip confirmation prompts
```

### Building the patched `libaims.so`

After patching, build FHI-aims as a **shared library** (Intel OneAPI for MPI + MKL is the tested configuration):

```bash
# Ensure the OneAPI environment is sourced (see Prerequisites above)
ulimit -s unlimited
mkdir build && cd build
# Prepare the Cmake file
cmake -DUSE_MPI=ON -DUSE_LIBXS=OFF -DBUILD_SHARED_LIBS=ON -C initial_cache.cmake ..
make -j 8
```

> **Note: shared library is required**
>
> AimsPy loads `libaims.so` via `ctypes` at runtime, so you **must**
> pass `-DBUILD_SHARED_LIBS=ON` to CMake. Without it, FHI-aims builds
> a static archive (`libaims.a` or `libaims.x`) which AimsPy cannot load.

The resulting `libaims.so` is what AimsPy loads. FHI-aims itself is **not** distributed with AimsPy and remains under its own licence agreement with the aims team, users must obtain the FHI-aims source code independently.

## Install from Source (for Developers)

```bash
git clone https://github.com/kYangLi/aimspy.git
# or, after forking:
# git clone https://github.com/<YourAccount>/aimspy.git

cd aimspy
uv pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `ruff`, and `black`. The `[docs]` extra (needed to build this documentation locally) pulls in Sphinx and the documentation theme:

```bash
uv pip install -e ".[docs]"
```

## Environment Variables

Integration tests and examples require `AIMSPY_TEST_AIMS_LIBPATH` to point at your patched `libaims.so`:

```bash
export AIMSPY_TEST_AIMS_LIBPATH=/path/to/FHI-aims-deeph/build/libaims.so
```

Optional:

- `AIMSPY_TEST_NPROC`: MPI process count for integration tests and examples (default: `8`).

For example, to run the H₂O baseline SCF example:

```bash
export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
export AIMSPY_TEST_NPROC=8
make run-from-scratch
```

## Next Steps

Once AimsPy is installed and `libaims.so` is patched and built, head to [Basic Usage](./basic_usage.md) for end-to-end examples covering baseline SCF, DeepH-format export, DeepH warmstart, and error recovery.
