"""Small MPI helpers shared by standalone integration tests."""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def synchronized_python_exceptions(comm) -> Iterator[None]:
    """Raise a Python failure on every rank after preserving its traceback.

    Every rank must enter the context.  The guarded block must not contain a
    collective that only some ranks can reach; use it around rank-local work,
    especially rank-zero-only filesystem validation and export.
    """
    local_error = None
    try:
        yield
    except Exception:
        local_error = traceback.format_exc()

    errors = comm.allgather(local_error)
    failures = [
        f"rank {failed_rank}:\n{error}"
        for failed_rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise RuntimeError(
            "Python exception synchronized across MPI ranks:\n" + "\n".join(failures)
        )
