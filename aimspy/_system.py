"""Encapsulated system-level helpers — the ONLY place `os` is imported.

All other aimspy modules use `pathlib.Path` exclusively.
This isolates the unavoidable `os` usage (chdir, environ) to one auditable file.
"""
from __future__ import annotations

import os as _os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def chdir_cm(target: Path) -> Iterator[Path]:
    """Context-manager: chdir into *target* on enter, restore on exit.

    Returns the previous cwd as a `Path` for caller convenience.
    """
    prev = Path.cwd()
    try:
        _os.chdir(str(target))
        yield prev
    finally:
        _os.chdir(str(prev))


def get_env(name: str, default: str | None = None) -> str | None:
    """Thin wrapper around `os.environ.get` — keeps `import os` local."""
    return _os.environ.get(name, default)
