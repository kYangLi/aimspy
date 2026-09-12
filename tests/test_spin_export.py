#!/usr/bin/env python
"""Spinful integration test: forward spin-collinear SCF → DeepH export →
cross-validation.

Runs a ``spin collinear`` SCF on MoS2 (initial moment 0.5), then:

1. Cross-validates the in-memory two-channel Hamiltonian
   ``calc.rs_hamiltonian`` (shape ``(2, n_ham)``) against the
   FHI-aims-written per-channel references ``rs_hamiltonian_up.out`` /
   ``rs_hamiltonian_dn.out`` (independent of the Python conversion path).
2. Exports H + S + H0 to ``tests/data/MoS2_spin/deeph_spin_out/`` via
   ``DeepHData.from_aimspy`` and validates the spinful on-disk contract
   (doubled chunk layout + ``spin_treatment: "collinear"``).
3. Reads the export back (``from_directory`` → ``to_aimspy`` →
   ``to_aims_csr``) and closes the loop against the same FHI-aims
   references.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_spin_export.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from mpi4py import MPI

from aimspy import AimspyMatrix, Calculator, CalculatorConfig
from aimspy import DeepHData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2_spin"
DEEPH_OUT = DATA_DIR / "deeph_spin_out"

comm = MPI.COMM_WORLD
rank = comm.rank

_lib_env = os.environ.get("AIMSPY_TEST_AIMS_LIBPATH")
if not _lib_env:
    if rank == 0:
        print(
            "ERROR: AIMSPY_TEST_AIMS_LIBPATH environment variable not set.\n"
            "  Export the path to your patched libaims.so before running:\n"
            "    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so",
            file=sys.stderr,
        )
    comm.Abort(1)
LIB_PATH = Path(_lib_env)


def _info(msg):
    if rank == 0:
        print(msg)


def _ok(name, condition, detail=""):
    tag = "OK " if condition else "FAIL"
    _info(f"  {tag}  {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


# =============================================================================
# Step 1: Spinful forward SCF
# =============================================================================
_info("=" * 60)
_info("Step 1: Spinful (spin collinear) forward SCF")
_info("=" * 60)

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_spin_export.out"),
    log_level="INFO",
    capture_initial_hamiltonian=True,
    capture_overlap=True,  # exercise the export_ovlp callback on a spinful system
)
calc = Calculator(config)
all_ok = True

try:
    calc.do(comm=comm, work_dir=DATA_DIR)

    if rank == 0:
        # -- 1a. Runtime spin contract --
        _info(f"  info.n_spin = {calc.info.n_spin} (expected 2)")
        all_ok &= _ok("info.n_spin == 2", calc.info.n_spin == 2)

        H = calc.rs_hamiltonian  # (2, n_ham)
        H_aimspy = calc.hamiltonian  # stacked blocks
        S_aimspy = calc.overlap
        h_init_aimspy = calc.initial_hamiltonian
        structure = calc.structure

        _info(f"  H shape: {H.shape}, max|H| = {np.max(np.abs(H)):.6e} Hartree")
        all_ok &= _ok("rs_hamiltonian shape (2, n_ham)", H.shape[0] == 2)

        all_ok &= _ok("calc.hamiltonian is stacked (n_spin=2)", H_aimspy.n_spin == 2)
        _info(f"  H blocks: {H_aimspy.n_pairs} pairs, n_spin={H_aimspy.n_spin}")
        all_ok &= _ok(
            "calc.overlap (export_ovlp callback) is spin-independent (n_spin=1)",
            S_aimspy.n_spin == 1,
        )
        # The callback-captured overlap must agree with the CSR fallback
        # path (rs_overlap) — both derive from the same Fortran data, so
        # this cross-checks the export_ovlp wiring under spin collinear.
        S_fallback = AimspyMatrix.from_aims_csr(
            calc.rs_overlap.reshape(1, -1), calc.csr_descr, structure
        )
        s_diff = 0.0
        if S_aimspy.blocks:
            assert set(S_aimspy.blocks) == set(S_fallback.blocks)
            s_diff = max(
                float(np.max(np.abs(S_aimspy.blocks[k] - S_fallback.blocks[k])))
                for k in S_aimspy.blocks
            )
        _info(f"  overlap callback vs CSR fallback: max|dS| = {s_diff:.2e}")
        all_ok &= _ok("overlap callback path matches CSR fallback", s_diff < 1e-12)
        all_ok &= _ok(
            "calc.initial_hamiltonian is stacked (n_spin=2)",
            h_init_aimspy.n_spin == 2,
        )

        # =================================================================
        # Step 2: Cross-validate against FHI-aims per-channel references
        # =================================================================
        _info("")
        _info("=" * 60)
        _info("Step 2: Cross-validate vs rs_hamiltonian_up/dn.out (Fortran)")
        _info("=" * 60)

        ref_up = np.loadtxt(DATA_DIR / "rs_hamiltonian_up.out", dtype=np.float64)
        ref_dn = np.loadtxt(DATA_DIR / "rs_hamiltonian_dn.out", dtype=np.float64)
        ref_up = ref_up.reshape(1, -1) if ref_up.ndim == 1 else ref_up
        ref_dn = ref_dn.reshape(1, -1) if ref_dn.ndim == 1 else ref_dn
        ref_H = np.vstack([ref_up, ref_dn])  # (2, n_ham)
        csr = calc.csr_descr
        trim = csr.n_ham_size - 1

        _info(f"  ref_up shape: {ref_up.shape}, ref_dn shape: {ref_dn.shape}")
        diff_up = np.max(np.abs(H[0, :trim] - ref_H[0, :trim]))
        diff_dn = np.max(np.abs(H[1, :trim] - ref_H[1, :trim]))
        _info(f"  max|H_up - ref_up| = {diff_up:.2e} Hartree")
        _info(f"  max|H_dn - ref_dn| = {diff_dn:.2e} Hartree")
        all_ok &= _ok(
            "alpha channel vs rs_hamiltonian_up.out",
            diff_up < 1e-8,
            f"max|diff|={diff_up:.2e}",
        )
        all_ok &= _ok(
            "beta channel vs rs_hamiltonian_dn.out",
            diff_dn < 1e-8,
            f"max|diff|={diff_dn:.2e}",
        )
        _info(
            f"  channel difference max|H_up - H_dn| = "
            f"{np.max(np.abs(H[0, :trim] - H[1, :trim])):.2e} Hartree"
        )

        # =================================================================
        # Step 3: Export to deeph_spin_out/
        # =================================================================
        _info("")
        _info("=" * 60)
        _info(f"Step 3: Export to {DEEPH_OUT} (spinful from_aimspy)")
        _info("=" * 60)

        dd = DeepHData.from_aimspy(
            structure,
            hamiltonian=H_aimspy,
            overlap=S_aimspy,
            initial_hamiltonian=h_init_aimspy,
        )
        _info(f"  DeepHData: {dd}")
        _info(f"  _spinful: {dd._spinful}")
        all_ok &= _ok("DeepHData._spinful", dd._spinful is True)

        # Doubled chunk layout: rows == 2 * per-atom orbital counts.
        counts = np.array(
            [
                sum(2 * ell + 1 for ell in dd.elements_orbital_map[sym])
                for sym in dd.atom_symbols
            ]
        )
        ap = dd.atom_pairs
        expected_rows = 2 * counts[ap[:, 3]]
        all_ok &= _ok(
            "chunk_shapes rows doubled (2*n_orb_i)",
            np.array_equal(dd.chunk_shapes[:, 0], expected_rows),
        )
        if dd._ovlp_chunk_shapes is not None:
            all_ok &= _ok(
                "overlap layout standard (_ovlp_chunk_*)",
                np.array_equal(dd._ovlp_chunk_shapes[:, 0], counts[ap[:, 3]]),
            )
        else:
            all_ok &= _ok("_ovlp_chunk_shapes present", False)

        if DEEPH_OUT.exists():
            shutil.rmtree(DEEPH_OUT)  # no stale h5 files from prior runs
        DEEPH_OUT.mkdir(parents=True, exist_ok=True)
        dd.save(DEEPH_OUT)
        _info("  Saved.")

        # =================================================================
        # Step 4: On-disk contract + read-back loop
        # =================================================================
        _info("")
        _info("=" * 60)
        _info("Step 4: On-disk contract + read-back cross-validation")
        _info("=" * 60)

        # -- 4a. info.json --
        with open(DEEPH_OUT / "info.json") as f:
            info_json = json.load(f)
        all_ok &= _ok("info.json spinful == true", info_json["spinful"] is True)
        all_ok &= _ok(
            "info.json spin_treatment == collinear",
            info_json.get("spin_treatment") == "collinear",
        )

        # -- 4b. h5 layout --
        with h5py.File(DEEPH_OUT / "hamiltonian.h5") as f:
            all_ok &= _ok(
                "hamiltonian.h5 chunk_shapes rows == 2*n_orb_i",
                np.array_equal(f["chunk_shapes"][:, 0], expected_rows),
            )
        with h5py.File(DEEPH_OUT / "overlap.h5") as f:
            all_ok &= _ok(
                "overlap.h5 chunk_shapes rows == n_orb_i (standard)",
                np.array_equal(f["chunk_shapes"][:, 0], counts[ap[:, 3]]),
            )

        # -- 4c. Full read-back loop vs the Fortran references --
        dd2 = DeepHData.from_directory(DEEPH_OUT)
        all_ok &= _ok("from_directory _spinful", dd2._spinful is True)
        H_back = dd2.to_aimspy(structure)
        all_ok &= _ok("to_aimspy n_spin == 2", H_back.n_spin == 2)
        H_csr = H_back.to_aims_csr(csr, structure)  # (2, n_ham)
        rt_diff = np.max(np.abs(H_csr[:, :trim] - ref_H[:, :trim]))
        _info(f"  read-back max|H_csr - ref| = {rt_diff:.2e} Hartree")
        all_ok &= _ok(
            "full loop (DeepH→aimspy→CSR) vs Fortran refs",
            rt_diff < 1e-8,
            f"max|diff|={rt_diff:.2e}",
        )

finally:
    calc.close()
    comm.Barrier()

# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    _info("")
    _info("=" * 60)
    if all_ok:
        _info("SPIN EXPORT TEST PASSED — all cross-validation OK")
    else:
        _info("SPIN EXPORT TEST FAILED — see failures above")
    _info("=" * 60)
    if not all_ok:
        sys.exit(1)
