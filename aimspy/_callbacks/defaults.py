"""Private — default callback implementations (Python-friendly signatures).

Each class here is a `DefaultCallback` subclass mapped to one `CallbackSpec`.
Users can replace any default by registering their own callable.
"""
from __future__ import annotations

import logging

import numpy as np

from .base import DefaultCallback
from .modifiers import HamiltonianModifier

_log = logging.getLogger(__name__)


# =========================================================================
# get_descr — capture CSR matrix descriptor
# =========================================================================
class GetDescrDefault(DefaultCallback):
    """Wires ``aux['descr']`` with a ``CsrMatrixDescriptor``.

    The actual descriptor population from the raw C struct is handled
    by the get_descr wrapper in ``_callbacks/base.py`` (since it needs
    access to the raw ``descr_ptr`` ctypes parameter, which is not
    passed through to the Python‑level callback).

    This class exists so that ``register_default`` has a valid spec
    implementation to instantiate.
    """

    def __call__(self, aux: dict) -> None:
        pass  # descriptor already populated by the wrapper


# =========================================================================
# export_h0 — capture initial Hamiltonian H0
# =========================================================================
class ExportH0Default(DefaultCallback):
    """Captures H0 into ``aux['init_hamiltonian']`` (or ``aux['history']`` list)."""

    def __call__(self, aux: dict, h0: np.ndarray,
                 n_ham: int, n_spin: int) -> None:
        keep_history = aux.get('keep_history', False)
        if keep_history:
            history = aux.setdefault('history', [])
            history.append(h0.copy())
        else:
            aux['init_hamiltonian'] = h0.copy()

    @classmethod
    def make_aux(cls, **kwargs) -> dict:
        d = {'keep_history': False}
        val = kwargs.pop('value', None)
        if isinstance(val, str) and val.strip().lower() == 'history':
            d['keep_history'] = True
        d.update(kwargs)
        return d


# =========================================================================
# modify_h0 — modify / inject Hamiltonian (strategy dispatch)
# =========================================================================
class ModifyH0Default(DefaultCallback):
    """Dispatches to the strategy stored in ``aux['strategy']``.

    Default strategy (if ``aux['strategy']`` is None) is **replace**.
    """

    def __call__(self, aux: dict, h0: np.ndarray,
                 n_ham: int, n_spin: int) -> None:
        strategy = aux.get('strategy')
        if strategy is None:
            strategy = HamiltonianModifier.replace
        strategy(h0, n_ham, n_spin, aux)

    @classmethod
    def make_aux(cls, **kwargs) -> dict:
        base = {'strategy': None}
        base.update(kwargs)
        return base


# =========================================================================
# python_func — generic Python hook
# =========================================================================
class PythonFuncDefault(DefaultCallback):
    """Invokes a user‑provided callable stored in ``aux['hook_fn']``."""

    def __call__(self, aux: dict) -> None:
        hook = aux.get('hook_fn')
        if hook is None:
            _log.debug("python_func default: no hook_fn in aux, skipping")
            return
        try:
            hook(aux)
        except Exception as exc:
            _log.error("python_func hook raised %s", exc, exc_info=True)
