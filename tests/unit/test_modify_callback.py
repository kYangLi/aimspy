"""Unit tests for the ``modify_h0`` callback chain (spinful + guards).

Exercises the full 8-line wiring in ``Calculator._wire_callbacks`` —
``python_func`` (external source conversion) followed by
``modify_h0`` (from_aims_csr → _apply_strategy → to_aims_csr → memmove) —
by calling the registered Python closures directly, without Fortran.

The closures are recovered from ``CallbackManager._wrapped[name][1]``,
which holds the original Python function alongside the ctypes wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aimspy import (
    AimspyMatrix,
    AimspyStructure,
    Calculator,
    CalculatorConfig,
    CalcState,
    CsrMatrixDescriptor,
    Strategy,
)
from aimspy._callbacks.base import CallbackManager
from aimspy._binding.prototypes import BindingLib
from aimspy._exceptions import AimspyConfigError

SENTINEL = np.iinfo(np.int32).max
N_BASIS = 4
N_ENTRIES_PER_CELL = N_BASIS * N_BASIS
N_HAM = 2 * N_ENTRIES_PER_CELL + 1  # ±x cells + trash slot


class _FakeCDLL:
    """Mimics ``ctypes.CDLL`` exposing only the register symbols."""

    def __init__(self, symbols):
        for name in symbols:
            setattr(self, name, lambda *a, **k: None)


def _make_structure() -> AimspyStructure:
    return AimspyStructure(
        n_atoms=2,
        n_basis=N_BASIS,
        n_spin=2,
        n_periodic=3,
        lattice=np.eye(3, dtype=np.float64),
        atom_symbols=["A", "B"],
        atom_coords=np.zeros((2, 3), dtype=np.float64),
        basis_atom=np.array([0, 0, 1, 1], dtype=np.int32),
        basis_l=np.array([0, 1, 0, 1], dtype=np.int32),
        basis_m=np.array([0, 1, 0, -1], dtype=np.int32),
    )


def _make_descriptor(n_spin: int) -> CsrMatrixDescriptor:
    n_cells = 3  # +x, −x, sentinel
    cell_idx = np.zeros((3, n_cells), dtype=np.int32)
    cell_idx[:, 0] = (1, 0, 0)
    cell_idx[:, 1] = (-1, 0, 0)
    cell_idx[:, 2] = SENTINEL
    row_mx_idx = np.zeros((N_BASIS, n_cells, 2), dtype=np.int32)
    for ic in range(2):
        for ib in range(N_BASIS):
            start = ic * N_ENTRIES_PER_CELL + ib * N_BASIS + 1
            row_mx_idx[ib, ic] = (start, start + N_BASIS - 1)
    col_mx_idx = np.zeros((N_HAM,), dtype=np.int32)
    for ic in range(2):
        for ib in range(N_BASIS):
            base = ic * N_ENTRIES_PER_CELL + ib * N_BASIS
            col_mx_idx[base : base + N_BASIS] = np.arange(1, N_BASIS + 1)
    return CsrMatrixDescriptor(
        n_basis=N_BASIS,
        n_spin=n_spin,
        n_cells=n_cells,
        n_ham_size=N_HAM,
        cell_idx=cell_idx,
        row_mx_idx=row_mx_idx,
        col_mx_idx=col_mx_idx,
    )


def _make_values(seed: int = 7):
    """Hermitian-consistent (2, N_HAM) CSR values (−x cell = +x transpose)."""
    rng = np.random.default_rng(seed)
    m_up = rng.standard_normal((N_BASIS, N_BASIS))
    m_dn = rng.standard_normal((N_BASIS, N_BASIS))
    h = np.zeros((2, N_HAM), dtype=np.float64)
    for ib in range(N_BASIS):
        for jb in range(N_BASIS):
            flat = ib * N_BASIS + jb
            off = N_ENTRIES_PER_CELL + flat
            h[0, flat] = m_up[ib, jb]
            h[1, flat] = m_dn[ib, jb]
            h[0, off] = m_up[jb, ib]
            h[1, off] = m_dn[jb, ib]
    return h


def _wire(strategy, stub_source):
    """Build a Calculator with manually injected state, run
    ``_wire_callbacks``, and return (fn_python, fn_modify, aux, descr)."""
    calc = Calculator(CalculatorConfig(lib_path="/fake/libaims.so"))
    calc._state = CalcState.INITED
    calc._structure = _make_structure()
    calc._rank = 0
    calc._modify = SimpleNamespace(
        strategy=strategy,
        factor=1.0,
        custom_fn=None,
        source=stub_source,
        deferred_fn=None,
    )
    fake_lib = _FakeCDLL(
        [
            "aimspy_register_get_descr_callback",
            "aimspy_register_python_callback",
            "aimspy_register_modify_h0_callback",
        ]
    )
    calc._cb_mgr = CallbackManager(BindingLib(fake_lib))
    calc._wire_callbacks()
    aux = calc._runtime_aux
    descr = _make_descriptor(2)
    aux["csr_descr"] = descr
    fn_python = calc._cb_mgr._wrapped["python_func"][1]
    fn_modify = calc._cb_mgr._wrapped["modify_h0"][1]
    return fn_python, fn_modify, aux, descr


def _wire_dhde(with_modify: bool):
    """Build a Calculator wired for the dHde callbacks (capture + optional
    modify) and return (fn_export, fn_modify, aux)."""
    calc = Calculator(
        CalculatorConfig(
            lib_path="/fake/libaims.so", capture_first_order_hamiltonian=True
        )
    )
    calc._state = CalcState.INITED
    calc._structure = _make_structure()
    calc._rank = 0
    if with_modify:
        calc._modify_first_order = SimpleNamespace(
            strategy=Strategy.REPLACE, source=None, deferred_fn=None
        )
    fake_lib = _FakeCDLL(
        [
            "aimspy_register_get_descr_callback",
            "aimspy_register_export_dHde_callback",
            "aimspy_register_modify_dHde_callback",
        ]
    )
    calc._cb_mgr = CallbackManager(BindingLib(fake_lib))
    calc._wire_callbacks()
    aux = calc._runtime_aux
    fn_export = calc._cb_mgr._wrapped["export_dHde"][1]
    fn_modify_dhde = calc._cb_mgr._wrapped["modify_dHde"][1] if with_modify else None
    return fn_export, fn_modify_dhde, aux


class TestModifyCallbackChain:
    def test_replace_double_channel(self):
        """REPLACE end-to-end: python_func injects the stacked source,
        modify_h0 rewrites both channels of h_init in place via memmove."""
        struct = _make_structure()
        descr = _make_descriptor(2)
        ext = AimspyMatrix.from_aims_csr(_make_values(seed=41), descr, struct)
        stub = SimpleNamespace(to_aimspy=lambda structure: ext)
        fn_python, fn_modify, aux, descr = _wire(Strategy.REPLACE, stub)

        h_init = _make_values(seed=7)
        fn_python(aux)
        fn_modify(aux, h_init, N_HAM, 2)

        expected = ext.to_aims_csr(descr, struct)
        assert h_init.shape == (2, N_HAM)
        np.testing.assert_array_equal(h_init, expected)
        # The beta channel is genuinely written (n_bytes = n_ham * 2 * 8).
        assert not np.allclose(h_init[1], h_init[0])

    def test_add_double_channel(self):
        """ADD end-to-end: both channels become live + external."""
        struct = _make_structure()
        descr = _make_descriptor(2)
        ext = AimspyMatrix.from_aims_csr(_make_values(seed=41), descr, struct)
        stub = SimpleNamespace(to_aimspy=lambda structure: ext)
        fn_python, fn_modify, aux, descr = _wire(Strategy.ADD, stub)

        h_init = _make_values(seed=7)
        before = h_init.copy()
        fn_python(aux)
        fn_modify(aux, h_init, N_HAM, 2)

        expected = before + ext.to_aims_csr(descr, struct)
        np.testing.assert_allclose(h_init, expected, rtol=0, atol=1e-14)

    def test_guard_spinless_source_into_spinful(self):
        """A spinless source on a spin-polarized live matrix raises inside
        the modify_h0 closure (previously: silent zero-fill)."""
        struct = _make_structure()
        descr = _make_descriptor(1)
        ext = AimspyMatrix.from_aims_csr(_make_values(seed=41)[:1], descr, struct)
        stub = SimpleNamespace(to_aimspy=lambda structure: ext)
        fn_python, fn_modify, aux, _ = _wire(Strategy.REPLACE, stub)

        h_init = _make_values(seed=7)
        fn_python(aux)
        with pytest.raises(AimspyConfigError, match="n_spin"):
            fn_modify(aux, h_init, N_HAM, 2)


class TestDhdeSpinfulGuards:
    """The dHde callbacks reject spin-polarized systems instead of
    silently dropping (capture) or zeroing (modify) the beta channel —
    the same class of guard as the modify_h0 cross-spin check."""

    def test_export_dHde_spinful_rejected(self):
        fn_export, _, _ = _wire_dhde(with_modify=False)
        dHde = np.zeros((2, N_HAM, 3), dtype=np.float64)
        with pytest.raises(AimspyConfigError, match="export_dHde"):
            fn_export({}, dHde, N_HAM, 3, 2, 0)
        # Serial mode signature (n_dir=1) is guarded identically.
        with pytest.raises(AimspyConfigError, match="beta channel"):
            fn_export({}, dHde[:, :, :1], N_HAM, 1, 2, 1)

    def test_export_dHde_spinless_not_rejected_by_guard(self):
        fn_export, _, _ = _wire_dhde(with_modify=False)
        dHde = np.zeros((1, N_HAM, 3), dtype=np.float64)
        # n_spin=1: the guard passes; with no csr_descr in aux the closure
        # returns early — no exception.
        fn_export({}, dHde, N_HAM, 3, 1, 0)

    def test_modify_dHde_spinful_rejected(self):
        _, fn_modify, _ = _wire_dhde(with_modify=True)
        with pytest.raises(AimspyConfigError, match="modify_dHde"):
            fn_modify({}, 0, N_HAM, 3, 2, 0)
        with pytest.raises(AimspyConfigError, match="beta channel"):
            fn_modify({}, 0, N_HAM, 1, 2, 2)

    def test_modify_dHde_spinless_not_rejected_by_guard(self):
        _, fn_modify, _ = _wire_dhde(with_modify=True)
        # n_spin=1: the guard passes; with no modify_first_order/csr_descr
        # in aux the closure returns early — no exception.
        fn_modify({}, 0, N_HAM, 3, 1, 0)
