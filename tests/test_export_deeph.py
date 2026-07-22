#!/usr/bin/env python
"""Forward SCF → DeepH export → cross-validation.

Runs a standard (non-warmstart) SCF, then exports H + S + H0 to
``tests/data/MoS2/deeph_out/`` via ``DeepHData.from_aimspy`` (no template).

Then cross-validates the exported data against the in-memory aimspy
matrices and the FHI-aims ``rs_hamiltonian.out`` reference.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_export_deeph.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig
from aimspy.interface.deeph import DeepHData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2"
DEEPH_OUT = DATA_DIR / "deeph_out"

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
# Step 1: Forward SCF
# =============================================================================
_info("=" * 60)
_info("Step 1: Forward SCF (capture H + S + H_init)")
_info("=" * 60)

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_export.out"),
    log_level="INFO",
    capture_initial_hamiltonian=True,
)
calc = Calculator(config)

try:
    calc.do(comm=comm, work_dir=DATA_DIR)

    if rank == 0:
        H_aimspy = calc.hamiltonian
        S_aimspy = calc.overlap
        h_init_aimspy = calc.initial_hamiltonian
        structure = calc.structure

        _info(f"  H:       {H_aimspy.n_pairs} pairs")
        _info(f"  S:       {S_aimspy.n_pairs} pairs")
        _info(f"  H_init:  {h_init_aimspy.n_pairs} pairs")
        _info(f"  structure: {structure.n_atoms} atoms, {structure.n_basis} basis")

        # =================================================================
        # Step 2: Export to deeph_out/
        # =================================================================
        _info("")
        _info("=" * 60)
        _info(f"Step 2: Export to {DEEPH_OUT} (from_aimspy, no template)")
        _info("=" * 60)

        dd = DeepHData.from_aimspy(
            structure,
            hamiltonian=H_aimspy,
            overlap=S_aimspy,
            initial_hamiltonian=h_init_aimspy,
        )
        _info(f"  DeepHData: {dd}")
        _info(f"  n_basis: {dd.n_basis}")
        _info(
            f"  overlap_entries: {'present' if dd.overlap_entries is not None else 'None'}"
        )
        _info(
            f"  initial_hamiltonian_entries: {'present' if dd.initial_hamiltonian_entries is not None else 'None'}"
        )

        DEEPH_OUT.mkdir(parents=True, exist_ok=True)
        dd.save(DEEPH_OUT)
        _info("  Saved.")

        # =================================================================
        # Step 3: Cross-validation
        # =================================================================
        _info("")
        _info("=" * 60)
        _info("Step 3: Cross-validation")
        _info("=" * 60)

        all_ok = True
        csr = calc.csr_descr
        trim = csr.n_ham_size - 1

        # -- 3a. DeepH → aimspy → aims CSR vs rs_hamiltonian.out --
        _info("")
        _info("-- DeepH → aimspy → aims CSR vs rs_hamiltonian.out --")
        ref_H_txt = np.loadtxt(DATA_DIR / "rs_hamiltonian.out", dtype=np.float64)
        ref_H_txt = ref_H_txt.reshape(1, -1) if ref_H_txt.ndim == 1 else ref_H_txt

        H_back = dd.to_aimspy(structure)
        H_csr = H_back.to_aims_csr(csr, structure)
        csr_diff = np.max(np.abs(ref_H_txt[0, :trim] - H_csr[0, :trim]))
        all_ok &= _ok(
            "DeepH→aimspy→aims CSR vs rs_hamiltonian.out",
            csr_diff < 1e-10,
            f"max|diff|={csr_diff:.2e}",
        )

        # -- 3b. H_init roundtrip (DeepH → aimspy → aims CSR) --
        _info("")
        _info("-- H_init (initial Hamiltonian) --")
        if dd.initial_hamiltonian_entries is not None:
            h_init_ent = dd.initial_hamiltonian_entries
            h_ent = dd.entries
            _info(f"  H_init entries shape: {h_init_ent.shape}")
            _info(f"  H_init max|entries|: {np.max(np.abs(h_init_ent)):.4f} eV")
            _info(f"  H       max|entries|: {np.max(np.abs(h_ent)):.4f} eV")
            h_init_diff = np.max(np.abs(h_ent - h_init_ent))
            _info(f"  H_init vs H max|diff|: {h_init_diff:.4f} eV (should be > 0)")
            all_ok &= _ok(
                "H_init differs from H",
                h_init_diff > 0.1,
                f"diff={h_init_diff:.4f}",
            )

            # Cross-validate H_init via DeepH → aimspy → aims CSR
            dd_h_init_only = DeepHData(
                lattice=dd.lattice,
                atom_symbols=dd.atom_symbols,
                atom_coords=dd.atom_coords,
                elements_orbital_map=dd.elements_orbital_map,
                n_basis=dd.n_basis,
                atom_pairs=dd.atom_pairs,
                chunk_boundaries=dd.chunk_boundaries,
                chunk_shapes=dd.chunk_shapes,
                entries=dd.initial_hamiltonian_entries,
            )
            h_init_back = dd_h_init_only.to_aimspy(structure)
            h_init_csr = h_init_back.to_aims_csr(csr, structure)
            rt_diff = np.max(
                np.abs(
                    h_init_csr[0, :trim]
                    - h_init_aimspy.to_aims_csr(csr, structure)[0, :trim]
                )
            )
            all_ok &= _ok(
                "H_init roundtrip (DeepH→aimspy→aims CSR)",
                rt_diff < 1e-10,
                f"max|diff|={rt_diff:.2e}",
            )
        else:
            _info("  H_init entries: None")
            all_ok = False

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
        _info("EXPORT TEST PASSED — all cross-validation OK")
    else:
        _info("EXPORT TEST FAILED — see failures above")
    _info("=" * 60)
    if not all_ok:
        sys.exit(1)
