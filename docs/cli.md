# Command-line Interface

The `aimspy` command-line tool provides a single subcommand, `patch`,
for managing the bundled FHI-aims patch. It can also be invoked as
`python -m aimspy patch` (equivalent).

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
