# Command-line Interface

The `aimspy` command-line tool provides the subcommands `patch`
(manage the bundled FHI-aims patch), `viz-basis` and `viz-grid`
(offline visualization). It can also be invoked as
`python -m aimspy <subcommand>` (equivalent).

Full patching instructions (including prerequisites and build steps) live
in [Installation & Setup](./installation_and_setup.md#patching-fhi-aims).

## Usage

```bash
aimspy patch [SOURCE] [OPTIONS]
```

## Options

| Flag | Description |
|------|-------------|
| `SOURCE` | FHI-aims source directory (default: current dir) |
| `-v, --patch-version TEXT` | Patch version to apply (default: latest bundled) |
| `-l, --list` | List bundled patches and exit |
| `--check, --dry-run` | Dry-run only; do not modify the tree |
| `--uninstall` | Reverse the currently-detected patch |
| `--no-git` | Force `patch(1)` instead of `git apply` |
| `-y, --yes` | Skip confirmation prompts |

## Example workflows

```bash
aimspy patch --list                              # what versions are bundled?
aimspy patch --check /path/to/FHI-aims          # would the latest patch apply cleanly?
aimspy patch -v v0.1.0 /path/to/FHI-aims -y     # apply a specific version non-interactively
aimspy patch --uninstall /path/to/FHI-aims      # reverse whatever is currently applied
```

## Patch version detection

The CLI auto-detects the currently-applied patch version by reading a
`PATCH_VERSION` line that the patch itself writes into the source tree's
`Makefile`. This enables safe upgrade flows: when a new version is
requested, the CLI uninstalls the old one first (after confirmation, or
immediately with `-y`).

If the detected version's diff is not bundled with your AimsPy install,
the CLI will refuse to uninstall and point you to the recovery command.

## Troubleshooting

For common patch-related issues (failed dry-run, version mismatch, etc.),
see [Troubleshooting](./troubleshooting.md).

---

# aimspy viz-basis

Plot NAO radial basis functions from a `basis.h5` file (written by
`BasisData.save_h5`). One figure is produced per element in the file;
runtime-free (no libaims / MPI needed).

## Usage

```bash
aimspy viz-basis BASIS_H5 [OPTIONS]
```

## Options

| Flag | Description |
|------|-------------|
| `--kind [u|phi]` | Plot u(r) (default) or φ(r) = u(r)/r |
| `-j, --jobs INT` | Parallel worker processes (default 1; only with `-o`) |
| `--n-plot INT` | Interpolated points per curve (default 500) |
| `--split-l` | One panel per angular momentum l |
| `--logx` | Log-scale the x axis (spreads the log-grid sampling evenly) |
| `--no-grid` | Hide the log-grid rug markers |
| `--no-type` | Omit the function type from curve labels |
| `--bohr` | X axis in bohr instead of Angstrom |
| `--r-max FLOAT` | X-axis upper limit (plot units) |
| `-o, --output-dir PATH` | Save one figure per element here as `ELEMENT_basis.FMT` (default: show interactively) |
| `--fmt [png|pdf|svg]` | Figure format with `-o` (default png) |

## Example

```bash
aimspy viz-basis basis.h5 -o figs/ -j 4          # all elements, 4 workers
aimspy viz-basis basis.h5 --kind phi --split-l   # interactive, per-l panels
aimspy viz-basis basis.h5 --logx                 # log x axis: even log-grid spread
```

---

# aimspy viz-grid

Plot real-space grid data from an npz file (`GridData.save_npz`).

## Usage

```bash
aimspy viz-grid NPZ_PATH [FIELD] [OPTIONS]
```

`FIELD` defaults to `rho` (also: `delta_rho`, `vks`, `vks0`, `vxc`, ...).

## Options

| Flag | Description |
|------|-------------|
| `--mode [scatter|contour|radial]` | Plot mode (default scatter) |
| `--axis INT` | Slice-plane normal 0/1/2 (default 2) |
| `--center FLOAT` | Slice position along the axis (default 0) |
| `--width FLOAT` | Half-thickness of the accepted slab (default 1.0) |
| `--nx, --ny INT` | Contour mesh resolution (default 200) |
| `-l, --log` | Colour by log10(value) — positive fields |
| `--symlog` | Diverging sym-log scale — difference fields |
| `--linthresh FLOAT` | Linear half-width for `--symlog` (default 1e-3) |
| `--point-size FLOAT` | Scatter marker size (default 5.0) |
| `--levels INT` | Contour levels (default 60) |
| `--atom-index INT` | Radial mode: 0-based atom index (required) |
| `--bohr` | Use bohr instead of Angstrom |
| `--lin-y` | Radial mode: linear y axis |
| `-o, --output PATH` | Save the figure here (format from suffix) instead of showing it |

## Example

```bash
aimspy viz-grid grid.npz delta_rho --symlog --linthresh 1e-3 -o drho.png
aimspy viz-grid grid.npz rho --mode radial --atom-index 0
```
