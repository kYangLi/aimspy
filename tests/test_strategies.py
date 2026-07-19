#!/usr/bin/env python
"""Integration test: all four Strategy variants.

Tests Strategy.ADD, Strategy.SCALE, and Strategy.CUSTOM on MoS2.
Strategy.REPLACE is already covered by test_warmstart.py.

**Important**: FHI-aims is a global Fortran singleton — only one
init/finalize cycle per MPI process. Therefore each strategy variant
runs in a SEPARATE MPI invocation via subprocess. The controlling
Python process dispatches them and aggregates results.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    python tests/test_strategies.py            # dispatches 4 sub-MPI jobs
    mpiexec -np 8 python tests/test_strategies.py --strategy baseline  # single
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2"
NPROC = os.environ.get("AIMSPY_TEST_NPROC", "8")

_lib_env = os.environ.get("AIMSPY_TEST_AIMS_LIBPATH")
if not _lib_env:
    print(
        "ERROR: AIMSPY_TEST_AIMS_LIBPATH environment variable not set.\n"
        "  Export the path to your patched libaims.so before running:\n"
        "    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so",
        file=sys.stderr,
    )
    sys.exit(1)
LIB_PATH = _lib_env

ref_H = np.loadtxt(DATA_DIR / "rs_hamiltonian.out", dtype=np.float64)
ref_H = ref_H.reshape(1, -1) if ref_H.ndim == 1 else ref_H


def run_single(strategy: str) -> dict:
    """Run one strategy variant in a sub-MPI process.

    Returns dict with H (rs_hamiltonian), energy, ok.
    """
    env = os.environ.copy()
    env["AIMSPY_TEST_STRATEGY"] = strategy
    cmd = ["mpiexec", "-np", NPROC, sys.executable, __file__, "--child", strategy]

    print(f"  dispatching: {' '.join(cmd[:4])} ... {strategy}")
    r = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(DATA_DIR.parent.parent),
    )
    # Parse output: child writes "RESULT: energy=<float> ok=<bool>"
    energy = None
    ok = False
    for line in r.stdout.splitlines():
        if line.startswith("RESULT:"):
            parts = dict(p.split("=") for p in line[7:].split() if "=" in p)
            energy = float(parts.get("energy", "nan"))
            ok = parts.get("ok", "False") == "True"
    if r.returncode != 0:
        print(f"  [child {strategy}] CRASHED (signal {r.returncode})")
        if r.stderr:
            print(r.stderr[-500:])
    return {
        "strategy": strategy,
        "energy": energy,
        "ok": ok,
        "returncode": r.returncode,
    }


def run_children():
    """Only called when --child is passed — runs one strategy in MPI."""
    strategy = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    from mpi4py import MPI
    from aimspy import Calculator, CalculatorConfig, Strategy
    from aimspy.interface.deeph import DeepHData

    comm = MPI.COMM_WORLD
    rank = comm.rank
    deeph_dir = DATA_DIR / "deeph_warm"

    config = CalculatorConfig(
        lib_path=LIB_PATH,
        logfile=Path(f"aims_strategy_{strategy}.out"),
        log_level="INFO",
        capture_initial_hamiltonian=True,  # needed for CUSTOM/SOURCE strategies
    )
    calc = Calculator(config)

    # Configure modify
    if strategy == "baseline":
        pass  # no modify
    elif strategy == "add":
        dd = DeepHData.from_directory(deeph_dir)
        calc.modify_init_ham(source=dd, strategy=Strategy.ADD)
    elif strategy == "scale":
        calc.modify_init_ham(strategy=Strategy.SCALE, factor=0.5)
    elif strategy == "custom":

        def custom_diag(live, external, structure, aux):
            """Zero out off-diagonal blocks."""
            for key in list(live.blocks.keys()):
                R1, R2, R3, i_atom, j_atom = key
                if i_atom != j_atom or (R1, R2, R3) != (0, 0, 0):
                    live.blocks[key] = np.zeros_like(live.blocks[key])

        calc.modify_init_ham(strategy=Strategy.CUSTOM, custom_fn=custom_diag)
    else:
        print(f"Unknown strategy: {strategy}", file=sys.stderr)
        sys.exit(2)

    try:
        calc.do(comm=comm, work_dir=DATA_DIR)
        if rank == 0:
            H = calc.rs_hamiltonian
            E = calc.energy
            ok = np.allclose(H, ref_H, atol=1e-6)
            print(f"RESULT: energy={E:.6f} ok={ok} strategy={strategy}")
    finally:
        calc.close()


def main():
    if "--child" in sys.argv:
        run_children()
        return

    print("=" * 60)
    print("Strategy integration test (sub-MPI dispatch)")
    print(f"  nproc per strategy: {NPROC}")
    print("=" * 60)

    strategies = ["baseline", "add", "scale", "custom"]
    results = {}
    for s in strategies:
        print(f"\n--- {s} ---")
        r = run_single(s)
        results[s] = r
        print(f"  energy={r['energy']}, ok={r['ok']}, rc={r['returncode']}")

    # Compare: all strategies should converge to same ground-state H
    print()
    print("=" * 60)
    base_E = results["baseline"]["energy"]
    all_ok = True
    for s in strategies:
        r = results[s]
        if r["energy"] is None or r["returncode"] != 0:
            print(f"  FAIL  {s}: crashed or no result")
            all_ok = False
            continue
        E = r["energy"]
        e_diff = abs(E - base_E) if base_E else float("inf")
        e_ok = e_diff < 1e-4
        h_ok = r["ok"]
        tag = "OK  " if (e_ok and h_ok) else "FAIL"
        print(f"  {tag}  {s}: E={E:.6f} (Δ={e_diff:.2e}), H_match={h_ok}")
        if not (e_ok and h_ok):
            all_ok = False

    print()
    print("=" * 60)
    if all_ok:
        print("ALL STRATEGY TESTS PASSED")
    else:
        print("SOME STRATEGY TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
