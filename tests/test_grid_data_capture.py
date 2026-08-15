#!/usr/bin/env python
"""Integration test: capture_grid_data=True (per-rank real-space grid data).

Runs a plain LDA SCF with ``capture_grid_data=True``, then gathers the
per-rank grid subsets on rank 0 and verifies physical self-consistency:

  1. ``sum(partition_tab * rho)`` ~= n_electrons  (density conservation)
  2. ``vks - vks0`` is non-trivial
  3. ``index_atom`` within range (0-based)
  4. ``vxc = vks - vh`` is everywhere <= 0  (LDA exchange sign)
  5. ``sum(partition_tab * rho0) / (4*pi)`` ~= n_electrons (free-atom ref)
  6. npz save/load round-trip preserves the gathered dataset

Prerequisites:
    tests/data/MoS2_LDA/control.in  (xc pw-lda, SCF only)

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_grid_data_capture.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig, GridData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2_LDA"
OUT_NPZ = DATA_DIR / "grid_gathered.npz"

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

if not (DATA_DIR / "control.in").is_file():
    if rank == 0:
        print(f"ERROR: {DATA_DIR / 'control.in'} not found.", file=sys.stderr)
    sys.exit(1)

all_ok = True


def _info(msg):
    if rank == 0:
        print(msg, flush=True)


def check(name, condition, detail=""):
    global all_ok
    if rank == 0:
        tag = "OK  " if condition else "FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  — {detail}"
        print(line, flush=True)
    if not condition:
        all_ok = False


def main():
    # Remove stale gathered output from a previous run.
    if rank == 0 and OUT_NPZ.exists():
        OUT_NPZ.unlink()
    comm.Barrier()

    cfg = CalculatorConfig(
        lib_path=LIB_PATH,
        logfile=Path("aims_grid_capture.out"),
        log_level="WARNING",
        capture_grid_data=True,
    )
    calc = Calculator(cfg)
    calc.do(comm=comm, work_dir=DATA_DIR)

    # ---- per-rank assertions ----
    local = calc.grid_data
    if local is None:
        check("grid_data captured on this rank", False)
        calc.close()
        comm.Barrier()
        sys.exit(1)

    check(
        f"rank {rank}: grid_data captured",
        local is not None and local.n_full_points > 0,
        f"n={local.n_full_points}, n_spin={local.n_spin}",
    )
    check(
        f"rank {rank}: index_atom 0-based",
        local.index_atom.min() >= 0,
        f"min={local.index_atom.min()}",
    )

    # ---- gather to rank 0 ----
    gd = GridData.gather(local, comm, root=0)
    calc.close()
    comm.Barrier()

    if rank == 0:
        _info("=" * 60)
        _info("  GRID DATA CAPTURE (aimspy封装) VERIFY")
        _info("=" * 60)
        check("gather returned data on root", gd is not None)
        _info(f"  global grid points = {gd.n_full_points}")
        _info(f"  n_spin             = {gd.n_spin}")

        # 1. density conservation (n_spin=1: rho[0] is total density)
        n_elec = gd.integrated_electrons()
        check(
            "sum(partition_tab * rho) ~= 74",
            abs(n_elec - 74.0) < 0.5,
            f"got {n_elec:.6f}",
        )

        # 2. dV_KS non-trivial
        max_dvks = float(np.max(np.abs(gd.delta_vks)))
        check(
            "max|V_KS - V_KS_0| > 1e-3 Ha",
            max_dvks > 1e-3,
            f"got {max_dvks:.4e} Ha",
        )

        # 3. atom indices valid (MoS2: 3 atoms)
        check(
            "index_atom within [0, n_atoms)",
            gd.index_atom.min() >= 0 and gd.index_atom.max() < gd.n_atoms,
            f"range [0, {gd.index_atom.max()}], n_atoms={gd.n_atoms}",
        )

        # 4. LDA exchange potential sign: vxc <= 0 everywhere
        max_vxc = float(np.max(gd.vxc))
        check(
            "vxc = vks - vh <= 0 (LDA exchange sign)",
            max_vxc <= 1e-8,
            f"max(vxc) = {max_vxc:.4e} Ha",
        )

        # 5. free-atom reference integrates to n_electrons
        #    (rho0 IS rho_free — the 4*pi factor was removed at import)
        n_free = float(np.sum(gd.partition_tab * gd.rho0))
        check(
            "sum(partition_tab * rho0) ~= 74",
            abs(n_free - 74.0) < 0.5,
            f"got {n_free:.6f}",
        )

        # 5b. structure fields filled from in-memory runtime structure
        check(
            "atom_coords present, shape (n_atoms, 3)",
            gd.atom_coords is not None and gd.atom_coords.shape == (gd.n_atoms, 3),
            f"shape={None if gd.atom_coords is None else gd.atom_coords.shape}",
        )
        check(
            "atom_symbols / lattice present",
            gd.atom_symbols is not None
            and len(gd.atom_symbols) == gd.n_atoms
            and gd.lattice is not None,
            f"symbols={gd.atom_symbols}",
        )

        # 6. npz round-trip (incl. structure fields)
        gd.save_npz(OUT_NPZ)
        gd2 = GridData.load_npz(OUT_NPZ)
        same = (
            gd2.n_full_points == gd.n_full_points
            and np.array_equal(gd2.coords, gd.coords)
            and np.array_equal(gd2.vks, gd.vks)
            and np.array_equal(gd2.atom_coords, gd.atom_coords)
            and list(gd2.atom_symbols) == list(gd.atom_symbols)
            and np.array_equal(gd2.lattice, gd.lattice)
        )
        check("npz save/load round-trip", same, f"file={OUT_NPZ.name}")

        _info("=" * 60)
        _info("  RESULT: " + ("PASSED" if all_ok else "FAILED"))
        _info("=" * 60)

    comm.Barrier()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
