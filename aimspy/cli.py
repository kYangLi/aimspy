"""aimspy command-line interface.

Click-based.  Currently provides ``aimspy patch`` for managing the bundled
FHI-aims patch (apply / uninstall / dry-run / list).  The patch logic itself
lives in :mod:`aimspy._patches._apply`; this module is a thin front-end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from ._patches import _apply
from ._version import __version__


@click.group()
@click.version_option(__version__, prog_name="aimspy")
def main() -> None:
    """aimspy command-line tools."""


@main.command()
@click.argument(
    "source",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=".",
    required=False,
)
@click.option(
    "--patch-version",
    "-v",
    "version",
    default=None,
    help="Patch version to apply, e.g. v0.1.0. Default: latest bundled.",
)
@click.option(
    "--list", "-l", "list_", is_flag=True, help="List bundled patches and exit."
)
@click.option(
    "--check",
    "--dry-run",
    "check",
    is_flag=True,
    help="Dry-run only; do not modify the source tree.",
)
@click.option(
    "--uninstall",
    is_flag=True,
    help="Uninstall the currently-applied patch (reverse by detected version).",
)
@click.option("--no-git", is_flag=True, help="Force patch(1) instead of git apply.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts.")
def patch(
    source: Path,
    version: str | None,
    list_: bool,
    check: bool,
    uninstall: bool,
    no_git: bool,
    yes: bool,
) -> None:
    """Apply or uninstall the aimspy patch on an FHI-aims source tree.

    SOURCE defaults to the current directory.  When a patch is already
    detected (via the source tree's Makefile) and a new apply is requested,
    the currently-applied version is uninstalled first after confirmation.
    """
    if list_:
        _print_list()
        return

    source = source.resolve()
    try:
        if uninstall:
            _do_uninstall(source, check=check, no_git=no_git, yes=yes)
        else:
            target = _apply.find_patch(version) if version else _apply.latest_patch()
            _do_apply_flow(source, target, check=check, no_git=no_git, yes=yes)
    except (KeyError, RuntimeError, OSError, subprocess.SubprocessError) as e:
        click.echo(str(e), err=True)
        sys.exit(2)


def _print_list() -> None:
    patches = _apply.list_patches()
    if not patches:
        click.echo("No bundled patches.")
        return
    click.echo("Bundled patches:")
    for p in patches:
        click.echo(f"  {p.version}")


def _do_uninstall(source: Path, *, check: bool, no_git: bool, yes: bool) -> None:
    detected = _apply.detect_installed_version(source)
    if detected is None:
        click.echo(
            "No patch version detected in source/Makefile; nothing to uninstall."
        )
        return
    try:
        patch_info = _apply.find_patch(detected)
    except KeyError as e:
        click.echo(str(e), err=True)
        click.echo(
            f"Cannot uninstall applied version {detected}: its diff is not "
            f"bundled with this aimspy install.",
            err=True,
        )
        sys.exit(2)

    click.echo(f"Detected applied patch: {detected}")
    if not check and not yes:
        if not click.confirm(
            f"Uninstall (reverse) patch {detected} from {source}?", default=False
        ):
            click.echo("Aborted.")
            return

    r = _apply.do_apply(source, patch_info, reverse=True, check=check, no_git=no_git)
    _report(r, source, detected, action="uninstall")


def _do_apply_flow(
    source: Path,
    target: _apply.PatchInfo,
    *,
    check: bool,
    no_git: bool,
    yes: bool,
) -> None:
    detected = _apply.detect_installed_version(source)

    if detected is None:
        click.echo(f"No applied patch detected; applying {target.version}.")
        r = _apply.do_apply(source, target, check=check, no_git=no_git)
        _report(r, source, target.version, action="apply")
        if not r.ok:
            sys.exit(1)
        return

    if detected == target.version:
        click.echo(f"Detected applied patch version {detected} (same as requested).")
    else:
        click.echo(
            f"Detected applied patch version {detected}; "
            f"requested {target.version}."
        )

    if not check and not yes:
        if not click.confirm(
            f"Uninstall {detected} first, then apply {target.version}?",
            default=False,
        ):
            click.echo("Aborted without applying.")
            return
    else:
        click.echo(f"Proceeding: uninstall {detected} then apply {target.version}.")

    try:
        old = _apply.find_patch(detected)
    except KeyError as e:
        click.echo(str(e), err=True)
        click.echo(
            f"Cannot uninstall currently-applied {detected}: its diff is not "
            f"bundled. Aborting; source tree untouched.",
            err=True,
        )
        sys.exit(2)

    if check:
        r_un = _apply.do_apply(source, old, reverse=True, check=True, no_git=no_git)
        click.echo(
            f"[dry-run] uninstall {detected}: "
            f"{'OK' if r_un.ok else 'FAILED'} (via {r_un.backend})"
        )
        if not r_un.ok:
            click.echo(r_un.detail, err=True)
            sys.exit(1)
        click.echo(
            f"[dry-run] apply {target.version}: not validated "
            f"(tree still carries {detected}; apply runs after uninstall)"
        )
        click.echo(
            f"Dry-run OK: would uninstall {detected} then apply "
            f"{target.version} to {source}"
        )
        return

    r_un = _apply.do_apply(source, old, reverse=True, check=False, no_git=no_git)
    if not r_un.ok:
        click.echo(f"Uninstall of {detected} FAILED:", err=True)
        click.echo(r_un.detail, err=True)
        sys.exit(1)
    click.echo(f"Uninstalled {detected} from {source} via {r_un.backend}.")

    r_ap = _apply.do_apply(source, target, check=False, no_git=no_git)
    if not r_ap.ok:
        click.echo(
            f"Apply of {target.version} FAILED after uninstalling {detected}:",
            err=True,
        )
        click.echo(r_ap.detail, err=True)
        click.echo(
            f"Recover manually with: aimspy patch -v {detected} {source}",
            err=True,
        )
        sys.exit(1)
    click.echo(f"Applied {target.version} to {source} via {r_ap.backend}.")


def _report(r: _apply.ApplyResult, source: Path, version: str, *, action: str) -> None:
    past = {"apply": "Applied", "uninstall": "Uninstalled"}.get(
        action, action.capitalize()
    )
    if r.ok:
        if r.check:
            click.echo(
                f"[dry-run] {action} {version} on {source}: OK (via {r.backend})"
            )
        else:
            click.echo(f"{past} {version} on {source} via {r.backend}.")
    else:
        click.echo(f"{past} {version} FAILED (via {r.backend}):", err=True)
        if r.detail:
            click.echo(r.detail, err=True)
