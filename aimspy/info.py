"""Public — load AimspyInfo from the live aims runtime."""

from __future__ import annotations

from ._binding.prototypes import BindingLib
from ._exceptions import AimspyBindingError
from .data import AimspyInfo


def load_info(binding: BindingLib) -> AimspyInfo:
    """Call ``aimspy_get_info()`` and build a snapshot dataclass.

    Must be called after ``aimspy_init()`` (and preferably before
    ``aimspy_finalize()``).

    Parameters
    ----------
    binding : BindingLib
        The loaded libaims wrapper.

    Returns
    -------
    AimspyInfo
        Independent snapshot — safe to hold after ``aimspy_finalize()``.

    Raises
    ------
    AimspyBindingError
        If ``aimspy_get_info`` is not available in the loaded library or
        returns NULL.
    """
    ptr = binding.aimspy_get_info()
    if not ptr:
        raise AimspyBindingError("aimspy_get_info() returned NULL")
    return AimspyInfo.from_c(ptr.contents)
