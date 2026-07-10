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

> **Status:** `v0.0.1` is a minimal placeholder release for reserving
> the package name on PyPI. The frontend (Python object model for aims
> inputs), backend (ctypes binding), and test suite (DFT benchmark
> cases) are under active development and will follow in subsequent
> versions.

## Installation

<!-- TODO: pip install / editable / dev extras -->

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

<!-- TODO: Calculator API, configuration, callbacks, examples -->

## Development

<!-- TODO: contributing, tests, lint, build -->

## Licensing

AimSpy is released under **GPL-3.0-or-later** (see `LICENSE`).

FHI-aims itself is **not** distributed with AimSpy and remains under its
own licence agreement with the aims team. Users must obtain FHI-aims
source code independently.
