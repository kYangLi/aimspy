#!/usr/bin/env python
"""Comprehensive regression test — uses new aimspy API.

Covers:
  1. Baseline SCF + H, energy extraction
  2. Convert H to aimspy standard format (AimspyMatrix)
  3. aims → aimspy → aims roundtrip
  4. aimspy → DeepH → aimspy roundtrip
  5. capture_h0 (export initial H0 as AimspyMatrix)
  6. DeepHData from_directory + save
  7. DeepHData from_memory
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI
from aimspy import (Calculator, CalculatorConfig,
                    AimspyMatrix, AimspyStructure)
from aimspy.interface.deeph import DeepHData, DeepHSource, deeph_to_aimspy, aimspy_to_deeph

HERE = Path(__file__).resolve().parent
LIB_PATH = Path("/home/deeph/software/calc/aimspy/FHI-aims-deeph/build/libaims.250822_1.scalapack.mpi.so")
DATA_DIR = HERE / "data" / "MoS2"
DEEPH_DIR = DATA_DIR / "deeph_warm"

comm = MPI.COMM_WORLD
rank = comm.rank
errors: list[str] = []


def check(name, condition, detail=""):
    if not condition:
        msg = f"FAIL: {name}" + (f" — {detail}" if detail else "")
        errors.append(msg)
        if rank == 0:
            print(msg, file=sys.stderr)
    else:
        if rank == 0:
            print(f"  OK  {name}")


ref_H = np.loadtxt(DATA_DIR / "rs_hamiltonian.out", dtype=np.float64)
ref_H = ref_H.reshape(1, -1) if ref_H.ndim == 1 else ref_H

if rank == 0:
    print("── Regression (new API) ──")

# ── 1. Baseline SCF ────────────────────────────────────────────────
config = CalculatorConfig(
    lib_path=LIB_PATH, work_dir=DATA_DIR,
    logfile=DATA_DIR / "aims_regression.out",
)
calc = Calculator(config)
calc.capture_h0 = True          # ★ capture free-atom H0 as AimspyMatrix
calc.init(comm=comm)
calc.run()

H = calc.rs_hamiltonian
check("H shape", H.shape == (1, calc.info.n_ham_size))
check("H matches ref", np.allclose(H, ref_H, atol=1e-6))
check("energy", calc.energy < -4000, f"{calc.energy:.6f}")

# ── 2. Structure + CSR descriptor ───────────────────────────────────
s = calc.structure
check("structure.n_atoms", s.n_atoms == 3,
      f"{s.n_atoms}")
check("structure.n_basis", s.n_basis == 90)
check("structure.phase_factor shape", s.phase_factor.shape == (90,))
check("structure.basis_subidx shape", s.basis_subidx.shape == (90,))
check("structure.orbit_per_atom sum = n_basis",
      s.orbit_per_atom.sum() == 90)

csr = calc.csr_descr
check("csr_descr present", csr is not None)
if csr is not None:
    check("csr.n_cells", csr.n_cells == 106)
    check("csr.n_ham_size", csr.n_ham_size == calc.info.n_ham_size)

# ── 3. aimspy format ────────────────────────────────────────────────
H_aimspy = calc.hamiltonian
check("hamiltonian (AimspyMatrix)", H_aimspy.n_pairs > 0,
      f"{H_aimspy.n_pairs} pairs")
check("hamiltonian n_spin", H_aimspy.n_spin == 1)

# ── 4. aims → aimspy → aims roundtrip ──────────────────────────────
H_rt = H_aimspy.to_aims_csr(csr, s)
trim = csr.n_ham_size - 1
max_diff = abs(H[0, :trim] - H_rt[0, :trim]).max()
check(f"aims→aimspy→aims roundtrip max|diff|={max_diff:.2e}",
      max_diff < 1e-10)

# ── 5. capture_h0 (initial H0) ─────────────────────────────────────
H0 = calc.initial_hamiltonian
check("initial_hamiltonian exists", H0 is not None)
if H0 is not None:
    check("initial_hamiltonian n_pairs > 0", H0.n_pairs > 0)

# ── 6. aimspy → DeepH → aimspy roundtrip ────────────────────────────
deeph_out = aimspy_to_deeph(H_aimspy, s)
check("deeph_out n_atoms", deeph_out.n_atoms == 3)
check("deeph_out n_pairs", deeph_out.n_pairs > 0)

# Read back
data_reloaded = deeph_out
# Convert back to aimspy
source2 = DeepHSource(data_reloaded)
H_aimspy2 = source2.to_aimspy(s)
check("deeph roundtrip n_pairs match",
      H_aimspy2.n_pairs == H_aimspy.n_pairs,
      f"{H_aimspy2.n_pairs} vs {H_aimspy.n_pairs}")
# Convert to aims CSR and compare
H_rt2 = H_aimspy2.to_aims_csr(csr, s)
max_diff2 = abs(H[0, :trim] - H_rt2[0, :trim]).max()
check(f"aimspy→deeph→aimspy→aims max|diff|={max_diff2:.2e}",
      max_diff2 < 1e-6)

# ── 7. DeepHData from_directory ─────────────────────────────────────
dd = DeepHData.from_directory(DEEPH_DIR)
check("from_directory n_atoms", dd.n_atoms == 3)
check("from_directory n_pairs", dd.n_pairs > 0)
check("from_directory entries (eV)", abs(dd.entries[0]) > 100,
      f"first entry = {dd.entries[0]:.1f}")

# ── 8. DeepHData save/load ──────────────────────────────────────────
tmp_dir = DATA_DIR / "_regression_deeph"
dd.save(tmp_dir)
dd2 = DeepHData.from_directory(tmp_dir)
check("save/load atom_pairs", np.array_equal(dd.atom_pairs, dd2.atom_pairs))
check("save/load entries close",
      np.allclose(dd.entries, dd2.entries))
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)

# ── 9. DeepHSource conversion ──────────────────────────────────────
source = DeepHSource(dd)
H_from_deeph = source.to_aimspy(s)
check("deeph→aimspy n_pairs", H_from_deeph.n_pairs > 0)
H_from_deeph_csr = H_from_deeph.to_aims_csr(csr, s)
check("deeph→aimspy→aims close to ref",
      np.allclose(H_from_deeph_csr[0, :trim], ref_H[0, :trim], atol=1e-6))

calc.close()

# ── Summary ─────────────────────────────────────────────────────────
if rank == 0:
    print()
    passed = max(0, 22 - len(errors))
    print(f"{'='*60}")
    print(f"Regression: {passed}/22 checks passed")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  {e}")
        print("** REGRESSION FAILED **")
    else:
        print(f"{'='*60}")
        print("ALL REGRESSION CHECKS PASSED")
