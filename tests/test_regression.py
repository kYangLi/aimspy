#!/usr/bin/env python
"""Comprehensive regression test — uses new aimspy API.

Covers:
  1. Baseline SCF + H, energy extraction
  2. Convert H to aimspy standard format (AimspyMatrix)
  3. aims → aimspy → aims roundtrip
  4. aimspy → DeepH → aimspy roundtrip
  5. capture_h0 (export initial H0 as AimspyMatrix)
  6. DeepHData from_directory + save
  7. DeepHData from_aimspy (H+S+H0 export with template)
"""
from __future__ import annotations

import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig
from aimspy.interface.deeph import DeepHData, DeepHSource, aimspy_to_deeph

HERE = Path(__file__).resolve().parent
LIB_PATH = Path("/home/deeph/software/calc/aimspy/FHI-aims-deeph/build/libaims.250822_1.scalapack.mpi.so")
DATA_DIR = HERE / "data" / "MoS2"
DEEPH_DIR = DATA_DIR / "deeph_warm"

comm = MPI.COMM_WORLD
rank = comm.rank
errors: list[str] = []
n_checks = 0


def check(name, condition, detail=""):
    global n_checks
    n_checks += 1
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

# ── 1. Baseline SCF (all ranks) ──────────────────────────────────────
config = CalculatorConfig(
    lib_path=LIB_PATH, work_dir=DATA_DIR,
    logfile=DATA_DIR / "aims_regression.out",
)
calc = Calculator(config)
calc.capture_h0 = True
calc.init(comm=comm)

try:
    calc.run()

    # Rank-0-only data: rs_hamiltonian, hamiltonian, overlap, energy
    H = None
    H_aimspy = None
    S_aimspy = None
    H0 = None
    energy = 0.0
    if rank == 0:
        H = calc.rs_hamiltonian
        energy = calc.energy
        H_aimspy = calc.hamiltonian
        S_aimspy = calc.overlap
        H0 = calc.initial_hamiltonian

    # Shared data (available on all ranks)
    s = calc.structure
    csr = calc.csr_descr

    # ── 1b. Rank-0 checks on baseline SCF ───────────────────────────
    if rank == 0:
        check("H shape", H.shape == (1, calc.info.n_ham_size))
        check("H matches ref", np.allclose(H, ref_H, atol=1e-6))
        check("energy", energy < -4000, f"{energy:.6f}")

    # ── 2. Structure + CSR descriptor (all ranks) ────────────────────
    check("structure.n_atoms", s.n_atoms == 3, f"{s.n_atoms}")
    check("structure.n_basis", s.n_basis == 90)
    check("structure.phase_factor shape", s.phase_factor.shape == (90,))
    check("structure.basis_subidx shape", s.basis_subidx.shape == (90,))
    check("structure.orbit_per_atom sum = n_basis",
          s.orbit_per_atom.sum() == 90)

    check("csr_descr present", csr is not None)
    if csr is not None:
        check("csr.n_cells", csr.n_cells == 106)
        check("csr.n_ham_size", csr.n_ham_size == calc.info.n_ham_size)

    # ── 3-11. Rank-0-only section ────────────────────────────────────
    if rank == 0:
        # ── 3. aimspy format ─────────────────────────────────────────
        check("hamiltonian (AimspyMatrix)", H_aimspy.n_pairs > 0,
              f"{H_aimspy.n_pairs} pairs")
        check("hamiltonian n_spin", H_aimspy.n_spin == 1)

        # ── 4. aims → aimspy → aims roundtrip ────────────────────────
        H_rt = H_aimspy.to_aims_csr(csr, s)
        trim = csr.n_ham_size - 1
        max_diff = abs(H[0, :trim] - H_rt[0, :trim]).max()
        check(f"aims→aimspy→aims roundtrip max|diff|={max_diff:.2e}",
              max_diff < 1e-10)

        # ── 5. capture_h0 (initial H0) ───────────────────────────────
        check("initial_hamiltonian exists", H0 is not None)
        if H0 is not None:
            check("initial_hamiltonian n_pairs > 0", H0.n_pairs > 0)

        # ── 6. aimspy → DeepH → aimspy roundtrip ─────────────────────
        deeph_out = aimspy_to_deeph(H_aimspy, s)
        check("deeph_out n_atoms", deeph_out.n_atoms == 3)
        check("deeph_out n_pairs", deeph_out.n_pairs > 0)
        check("deeph_out overlap_entries is None",
              deeph_out.overlap_entries is None,
              "fresh aimspy→deeph should have no overlap")
        check("deeph_out n_basis == 90", deeph_out.n_basis == 90,
              f"n_basis={deeph_out.n_basis}")

        source2 = DeepHSource(deeph_out)
        H_aimspy2 = source2.to_aimspy(s)
        check("deeph roundtrip n_pairs match",
              H_aimspy2.n_pairs == H_aimspy.n_pairs,
              f"{H_aimspy2.n_pairs} vs {H_aimspy.n_pairs}")
        H_rt2 = H_aimspy2.to_aims_csr(csr, s)
        max_diff2 = abs(H[0, :trim] - H_rt2[0, :trim]).max()
        check(f"aimspy→deeph→aimspy→aims max|diff|={max_diff2:.2e}",
              max_diff2 < 1e-6)

        # ── 7. from_aimspy with S and H0 ─────────────────────────────
        dd_all = DeepHData.from_aimspy(s, H=H_aimspy, S=S_aimspy, H0=H0)
        check("from_aimspy(H+S+H0) n_atoms", dd_all.n_atoms == 3)
        check("from_aimspy(H+S+H0) n_pairs", dd_all.n_pairs > 0)
        check("from_aimspy n_basis == 90", dd_all.n_basis == 90,
              f"n_basis={dd_all.n_basis}")
        check("from_aimspy has overlap_entries",
              dd_all.overlap_entries is not None)
        check("from_aimspy has initial_hamiltonian_entries",
              dd_all.initial_hamiltonian_entries is not None)

        # ── 8. from_aimspy save/load roundtrip ───────────────────────
        tmp_dir2 = DATA_DIR / "_regression_full"
        dd_all.save(tmp_dir2)
        dd_reloaded = DeepHData.from_directory(tmp_dir2)
        check("save/load full H entries", np.allclose(dd_all.entries, dd_reloaded.entries))
        check("save/load full S entries", np.allclose(dd_all.overlap_entries, dd_reloaded.overlap_entries))
        check("save/load full H0 entries", np.allclose(dd_all.initial_hamiltonian_entries, dd_reloaded.initial_hamiltonian_entries))
        check("save/load n_basis", dd_reloaded.n_basis == dd_all.n_basis,
              f"{dd_reloaded.n_basis} vs {dd_all.n_basis}")

        with open(tmp_dir2 / "info.json", "r") as f:
            info_json = json.load(f)
        check("info.json atoms_quantity", info_json["atoms_quantity"] == 3)
        check("info.json orbits_quantity == 90", info_json["orbits_quantity"] == 90,
              f"got {info_json['orbits_quantity']}")
        check("info.json occupation > 0", info_json["occupation"] > 0,
              f"occupation={info_json['occupation']}")
        eom = info_json["elements_orbital_map"]
        check("elements_orbital_map has Mo", "Mo" in eom)
        check("elements_orbital_map has S", "S" in eom)
        check("Mo shells >= 4", len(eom.get("Mo", [])) >= 4,
              f"Mo shells: {eom.get('Mo', [])}")
        check("S shells >= 2", len(eom.get("S", [])) >= 2,
              f"S shells: {eom.get('S', [])}")
        shutil.rmtree(tmp_dir2, ignore_errors=True)

        # ── 9. from_aimspy with template ─────────────────────────────
        dd_templ = DeepHData.from_directory(DEEPH_DIR)
        dd_from_templ = DeepHData.from_aimspy(s, H=H_aimspy, template=dd_templ)
        check("from_aimspy(template) n_atoms", dd_from_templ.n_atoms == dd_templ.n_atoms)
        check("from_aimspy(template) atom_symbols match",
              list(dd_from_templ.atom_symbols) == list(dd_templ.atom_symbols))
        check("from_aimspy(template) elements_orbital_map match",
              dd_from_templ.elements_orbital_map == dd_templ.elements_orbital_map)
        check("from_aimspy(template) H entries shape matches",
              dd_from_templ.entries.shape[0] == dd_all.entries.shape[0])
        check("from_aimspy(template) n_basis from template",
              dd_from_templ.n_basis == dd_templ.n_basis,
              f"{dd_from_templ.n_basis} vs {dd_templ.n_basis}")

        # ── 10. DeepHData from_directory + save/load ─────────────────
        dd = DeepHData.from_directory(DEEPH_DIR)
        check("from_directory n_atoms", dd.n_atoms == 3)
        check("from_directory n_pairs", dd.n_pairs > 0)
        check("from_directory entries (eV)", abs(dd.entries[0]) > 100,
              f"first entry = {dd.entries[0]:.1f}")

        tmp_dir = DATA_DIR / "_regression_deeph"
        dd.save(tmp_dir)
        dd2 = DeepHData.from_directory(tmp_dir)
        check("save/load atom_pairs", np.array_equal(dd.atom_pairs, dd2.atom_pairs))
        check("save/load entries close", np.allclose(dd.entries, dd2.entries))
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── 11. DeepHSource conversion ───────────────────────────────
        source = DeepHSource(dd)
        H_from_deeph = source.to_aimspy(s)
        check("deeph→aimspy n_pairs", H_from_deeph.n_pairs > 0)
        H_from_deeph_csr = H_from_deeph.to_aims_csr(csr, s)
        check("deeph→aimspy→aims close to ref",
              np.allclose(H_from_deeph_csr[0, :trim], ref_H[0, :trim], atol=1e-6))

finally:
    calc.close()
    comm.Barrier()

# ── Summary (rank 0 only) ────────────────────────────────────────────
if rank == 0:
    print()
    passed = n_checks - len(errors)
    print(f"{'='*60}")
    print(f"Regression: {passed}/{n_checks} checks passed")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  {e}")
        print("** REGRESSION FAILED **")
    else:
        print(f"{'='*60}")
        print("ALL REGRESSION CHECKS PASSED")
