"""Private — patch discovery, version detection, and apply/reverse logic.

Pure-Python, no click dependency; safe to unit-test and reuse from the CLI
or any other front-end.

Version contract
----------------
Bundled diff files live next to this module under ``aimspy/_patches/`` and
follow the name pattern ``aimspy-patch_v<major>.<minor>.<patch>.diff``.

Once a patch is applied to an FHI-aims source tree, the tree's root
``Makefile`` carries a line ``PATCH_VERSION := v<major>.<minor>.<patch>``;
that line is part of the diff itself, so it appears on apply and disappears
on reverse.  ``detect_installed_version`` reads it back to identify which
version is currently applied.
"""

from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterator

_VERSION_RE = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
_MAKEFILE_VERSION_RE = re.compile(
    r"^\s*PATCH_VERSION\s*[:?+]?=\s*(v\d+\.\d+\.\d+)", re.MULTILINE
)
_PATCH_SUFFIX = ".diff"


@dataclass(frozen=True)
class PatchInfo:
    """One bundled patch file.

    ``resource`` is an importlib Traversable; materialize it with
    :func:`materialize` (or ``resources.as_file``) before touching the bytes,
    so zip-installed packages work too.
    """

    version: str
    version_tuple: tuple[int, int, int]
    resource: resources.abc.Traversable


@dataclass
class ApplyResult:
    """Outcome of an apply / reverse / dry-run attempt."""

    ok: bool
    backend: str
    action: str
    check: bool
    message: str
    detail: str = ""


def _parse_version(name: str) -> tuple[int, int, int] | None:
    m = _VERSION_RE.search(name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def list_patches() -> list[PatchInfo]:
    """All bundled patches, sorted ascending by numeric version."""
    root = resources.files("aimspy").joinpath("_patches")
    out: list[PatchInfo] = []
    for child in root.iterdir():
        name = child.name
        if not name.endswith(_PATCH_SUFFIX):
            continue
        vt = _parse_version(name)
        if vt is None:
            continue
        out.append(
            PatchInfo(
                version=_VERSION_RE.search(name).group(0),
                version_tuple=vt,
                resource=child,
            )
        )
    out.sort(key=lambda p: p.version_tuple)
    return out


def latest_patch() -> PatchInfo:
    patches = list_patches()
    if not patches:
        raise RuntimeError("No bundled patches found in aimspy._patches")
    return patches[-1]


def find_patch(version: str) -> PatchInfo:
    """Look up a patch by version, accepting ``v0.1.0`` or ``0.1.0``."""
    if not version.startswith("v"):
        version = "v" + version
    for p in list_patches():
        if p.version == version:
            return p
    available = ", ".join(p.version for p in list_patches()) or "(none)"
    raise KeyError(f"Patch {version} not bundled. Available: {available}")


@contextmanager
def materialize(patch_info: PatchInfo) -> Iterator[Path]:
    """Yield a real filesystem Path for *patch_info*'s diff bytes."""
    with resources.as_file(patch_info.resource) as p:
        yield Path(p)


def detect_installed_version(source: Path) -> str | None:
    """Read ``source/Makefile`` and return the applied ``PATCH_VERSION``, or None."""
    makefile = source / "Makefile"
    if not makefile.is_file():
        return None
    try:
        text = makefile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _MAKEFILE_VERSION_RE.search(text)
    if not m:
        return None
    return m.group(1)


def is_git_repo(source: Path) -> bool:
    if not source.is_dir():
        return False
    r = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def do_apply(
    source: Path,
    patch_info: PatchInfo,
    *,
    reverse: bool = False,
    check: bool = False,
    no_git: bool = False,
) -> ApplyResult:
    """Apply, reverse, or dry-run *patch_info* onto *source*.

    Preference: ``git apply`` when *source* is a git repo and ``no_git`` is
    False; otherwise ``patch -p1``.  A real apply/reverse is always preceded
    by a dry-run (``--check`` / ``--dry-run``); when ``check=True`` the
    dry-run result is returned without touching the tree.

    ``patch(1)`` is always invoked with ``--batch`` and a closed stdin so it
    never blocks on an interactive prompt (the CLI must stay non-interactive).
    """
    action = "reverse" if reverse else "apply"
    flag = ["--reverse"] if reverse else []
    pflag = ["-R"] if reverse else []

    use_git = (not no_git) and is_git_repo(source)

    with materialize(patch_info) as patch_path:
        if use_git:
            dry = subprocess.run(
                ["git", "-C", str(source), "apply", "--check", *flag, str(patch_path)],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            if dry.returncode == 0:
                if check:
                    return ApplyResult(
                        True,
                        "git",
                        action,
                        True,
                        f"git apply --check passed ({action})",
                        "",
                    )
                real = subprocess.run(
                    ["git", "-C", str(source), "apply", *flag, str(patch_path)],
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                )
                ok = real.returncode == 0
                return ApplyResult(
                    ok,
                    "git",
                    action,
                    False,
                    f"git apply {'--reverse ' if reverse else ''}"
                    f"{'ok' if ok else 'FAILED'}",
                    real.stdout + real.stderr,
                )

        prev_flag = "-R " if reverse else ""
        dry = subprocess.run(
            [
                "patch",
                "-d",
                str(source),
                "-p1",
                "--batch",
                "--dry-run",
                *pflag,
                "-i",
                str(patch_path),
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if dry.returncode != 0:
            return ApplyResult(
                False,
                "patch",
                action,
                check,
                f"patch --dry-run FAILED ({action}); " f"apply may not be clean",
                dry.stdout + dry.stderr,
            )
        if check:
            return ApplyResult(
                True, "patch", action, True, f"patch --dry-run passed ({action})", ""
            )
        real = subprocess.run(
            [
                "patch",
                "-d",
                str(source),
                "-p1",
                "--batch",
                *pflag,
                "-i",
                str(patch_path),
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        ok = real.returncode == 0
        return ApplyResult(
            ok,
            "patch",
            action,
            False,
            f"patch {prev_flag}{'ok' if ok else 'FAILED'}",
            real.stdout + real.stderr,
        )
