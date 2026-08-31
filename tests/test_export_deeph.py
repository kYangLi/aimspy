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

import h5py
import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig
from aimspy import DeepHData
from aimspy.data import HARTREE_TO_EV

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
        forces_aimspy = calc.forces  # (n_atoms, 3) eV/Å, aims order
        stress_aimspy = calc.stress  # (3, 3) eV/Å³ or None
        energy_relative_hartree = calc.energy_free_relative

        _info(f"  H:       {H_aimspy.n_pairs} pairs")
        _info(f"  S:       {S_aimspy.n_pairs} pairs")
        _info(f"  H_init:  {h_init_aimspy.n_pairs} pairs")
        _info(f"  structure: {structure.n_atoms} atoms, {structure.n_basis} basis")
        if forces_aimspy is not None:
            _info(
                f"  forces:  shape={forces_aimspy.shape}, "
                f"max|F|={np.max(np.abs(forces_aimspy)):.4e} eV/Å"
            )
        else:
            _info("  forces:  None (compute_forces not set)")
        _info(f"  energy_relative:  {energy_relative_hartree:.6f} Hartree")
        _info(f"  stress:  {'present' if stress_aimspy is not None else 'None'}")

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
            force=forces_aimspy,
            energy=energy_relative_hartree,
            stress=stress_aimspy,
        )
        _info(f"  DeepHData: {dd}")
        _info(f"  n_basis: {dd.n_basis}")
        _info(
            f"  overlap_entries: {'present' if dd.overlap_entries is not None else 'None'}"
        )
        _info(
            f"  initial_hamiltonian_entries: {'present' if dd.initial_hamiltonian_entries is not None else 'None'}"
        )
        if dd.force is not None:
            _info(f"  force:   shape={dd.force.shape}, POSCAR order")
            _info(f"  energy_eV: {dd.energy_eV:.6f} eV")

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
            # The plain-text reference is rounded on output; the independent
            # in-memory H_init roundtrip below retains the stricter 1e-10 check.
            csr_diff < 1e-8,
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

        # -- 3c. force.h5 cross-validation --
        _info("")
        _info("-- force.h5 (force + energy) --")
        if forces_aimspy is not None and dd.force is not None:
            # force.h5 file exists on disk
            force_h5_path = DEEPH_OUT / "force.h5"
            all_ok &= _ok(
                "force.h5 file exists", force_h5_path.is_file(), str(force_h5_path)
            )
            if force_h5_path.is_file():
                with h5py.File(force_h5_path, "r") as f:
                    # Dataset checks
                    all_ok &= _ok(
                        "force.h5 has cell dataset",
                        "cell" in f and f["cell"].shape == (3, 3),
                    )
                    all_ok &= _ok(
                        "force.h5 has energy dataset",
                        "energy" in f and f["energy"].shape == (),
                    )
                    all_ok &= _ok(
                        "force.h5 has force dataset",
                        "force" in f and f["force"].shape == (structure.n_atoms, 3),
                    )
                    if stress_aimspy is None:
                        all_ok &= _ok(
                            "force.h5 uses zero stress when unavailable",
                            "stress" in f
                            and f["stress"].shape == (6,)
                            and np.array_equal(f["stress"][:], np.zeros(6)),
                        )
                    else:
                        all_ok &= _ok(
                            "force.h5 has stress dataset",
                            "stress" in f and f["stress"].shape == (6,),
                        )
                    # Attr checks
                    all_ok &= _ok(
                        "force.h5 has formula attr",
                        "formula" in f.attrs and f.attrs["formula"] == b"X3",
                    )
                    all_ok &= _ok(
                        "force.h5 has natoms attr",
                        "natoms" in f.attrs and int(f.attrs["natoms"]) == 3,
                    )
                    # Energy value: Hartree → eV
                    energy_eV_expected = energy_relative_hartree * HARTREE_TO_EV
                    energy_eV_disk = float(f["energy"][()])
                    all_ok &= _ok(
                        "force.h5 energy matches (Hartree→eV)",
                        abs(energy_eV_disk - energy_eV_expected) < 1e-6,
                        f"disk={energy_eV_disk:.6f} vs expected={energy_eV_expected:.6f}",
                    )
                    # Force values: reorder dd.force (POSCAR) → aims order, compare
                    old2new, _ = structure.build_atom_permutation()
                    force_back_to_aims = dd.force[old2new]
                    force_disk = f["force"][:]
                    force_disk_aims = force_disk[old2new]
                    force_diff = np.max(np.abs(force_disk_aims - forces_aimspy))
                    all_ok &= _ok(
                        "force.h5 force matches (POSCAR→aims reorder)",
                        force_diff < 1e-10,
                        f"max|diff|={force_diff:.2e}",
                    )
                    if stress_aimspy is not None:
                        stress_expected = stress_aimspy[
                            (0, 1, 2, 1, 0, 0), (0, 1, 2, 2, 2, 1)
                        ]
                        stress_diff = np.max(np.abs(f["stress"][:] - stress_expected))
                        all_ok &= _ok(
                            "force.h5 stress tensor→Voigt",
                            stress_diff < 1e-10,
                            f"max|diff|={stress_diff:.2e}",
                        )
        else:
            _info("  forces not available — skipping force.h5 checks")
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
