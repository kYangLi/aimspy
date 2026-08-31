#!/usr/bin/env python3
"""Benchmark AimsPy CSR/block matrix conversion without benchmark frameworks.

Each requested case is executed in a fresh Python subprocess.  The default
synthetic cases use one warmup and five measured samples; the stress case is
only run when selected explicitly with ``--tier stress``.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import aimspy  # noqa: E402
from aimspy import AimspyMatrix, AimspyStructure, CsrMatrixDescriptor  # noqa: E402


@dataclass(frozen=True)
class SyntheticProfile:
    n_basis: int
    active_cells: int
    nonzeros_per_row: int
    orbitals_per_atom: int = 10

    @property
    def n_entries(self) -> int:
        return self.n_basis * self.active_cells * self.nonzeros_per_row


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    matrix: np.ndarray
    descriptor: CsrMatrixDescriptor
    structure: AimspyStructure
    source: str
    seed: int | None


PROFILES: dict[str, SyntheticProfile] = {
    "tiny": SyntheticProfile(12, 3, 4, 4),
    "small": SyntheticProfile(90, 105, 10),
    "medium": SyntheticProfile(300, 64, 16),
    "large": SyntheticProfile(1000, 48, 24),
    "stress": SyntheticProfile(2000, 64, 32),
}
DEFAULT_TIERS = ("small", "medium", "large")


def _synthetic_case(name: str, seed: int) -> BenchmarkCase:
    profile = PROFILES[name]
    n_basis = profile.n_basis
    active_cells = profile.active_cells
    nnz = profile.nonzeros_per_row
    n_cells = active_cells + 1
    n_ham_size = profile.n_entries + 1

    cell_idx = np.zeros((3, n_cells), dtype=np.int32)
    cell_idx[0, :active_cells] = np.arange(1, active_cells + 1, dtype=np.int32)
    cell_idx[:, -1] = np.iinfo(np.int32).max

    row_mx_idx = np.zeros((n_basis, n_cells, 2), dtype=np.int32)
    starts = (
        np.arange(active_cells * n_basis, dtype=np.int64).reshape(active_cells, n_basis)
        * nnz
        + 1
    )
    row_mx_idx[:, :active_cells, 0] = starts.T
    row_mx_idx[:, :active_cells, 1] = (starts + nnz - 1).T

    row_numbers = np.tile(np.arange(n_basis, dtype=np.int64), active_cells)
    offsets = np.arange(nnz, dtype=np.int64)
    col_mx_idx = np.empty(n_ham_size, dtype=np.int32)
    col_mx_idx[:-1] = ((row_numbers[:, None] + offsets[None, :]) % n_basis + 1).ravel()
    col_mx_idx[-1] = 0

    orbitals_per_atom = profile.orbitals_per_atom
    basis_atom = np.arange(n_basis, dtype=np.int32) // orbitals_per_atom
    n_atoms = int(basis_atom[-1]) + 1
    structure = AimspyStructure(
        n_atoms=n_atoms,
        n_basis=n_basis,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3, dtype=np.float64),
        atom_symbols=["X"] * n_atoms,
        atom_coords=np.zeros((n_atoms, 3), dtype=np.float64),
        basis_atom=basis_atom,
        basis_l=np.zeros(n_basis, dtype=np.int32),
        basis_m=np.zeros(n_basis, dtype=np.int32),
    )
    descriptor = CsrMatrixDescriptor(
        n_basis=n_basis,
        n_spin=1,
        n_cells=n_cells,
        n_ham_size=n_ham_size,
        cell_idx=cell_idx,
        row_mx_idx=row_mx_idx,
        col_mx_idx=col_mx_idx,
    )
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((1, n_ham_size))
    matrix[0, -1] = 0.0
    return BenchmarkCase(name, matrix, descriptor, structure, "synthetic", seed)


def _metadata_value(lines: list[str], key: str) -> int:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return int(line.split(":", 1)[1])
    raise ValueError(f"rs_indices.out: missing {key!r}")


def _integer_tokens(lines: list[str]) -> np.ndarray:
    values: list[int] = []
    for line in lines:
        values.extend(int(value) for value in line.split())
    return np.asarray(values, dtype=np.int64)


def _parse_rs_indices(path: Path) -> CsrMatrixDescriptor:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    n_ham_size = _metadata_value(lines, "n_hamiltonian_matrix_size")
    n_cells = _metadata_value(lines, "n_cells_in_hamiltonian")
    n_basis = _metadata_value(lines, "n_basis")

    try:
        cell_header = lines.index("cell_index")
        start_header = lines.index("index_hamiltonian(1,:,:)")
        end_header = lines.index("index_hamiltonian(2,:,:)")
        column_header = lines.index("column_index_hamiltonian")
    except ValueError as exc:
        raise ValueError(f"{path}: missing a required section header") from exc

    cells = _integer_tokens(lines[cell_header + 1 : start_header])
    starts = _integer_tokens(lines[start_header + 1 : end_header])
    ends = _integer_tokens(lines[end_header + 1 : column_header])
    columns = _integer_tokens(lines[column_header + 1 :])
    expected_row_values = n_cells * n_basis
    if cells.size != n_cells * 3:
        raise ValueError(
            f"{path}: cell_index has {cells.size} values; expected {n_cells * 3}"
        )
    if starts.size != expected_row_values or ends.size != expected_row_values:
        raise ValueError(
            f"{path}: index_hamiltonian sections must each contain "
            f"{expected_row_values} values"
        )
    if columns.size != n_ham_size:
        raise ValueError(
            f"{path}: column_index_hamiltonian has {columns.size} values; "
            f"expected {n_ham_size}"
        )

    row_mx_idx = np.empty((n_basis, n_cells, 2), dtype=np.int32)
    row_mx_idx[:, :, 0] = starts.reshape(n_cells, n_basis).T
    row_mx_idx[:, :, 1] = ends.reshape(n_cells, n_basis).T
    return CsrMatrixDescriptor(
        n_basis=n_basis,
        n_spin=1,
        n_cells=n_cells,
        n_ham_size=n_ham_size,
        cell_idx=cells.reshape(n_cells, 3).T.astype(np.int32),
        row_mx_idx=row_mx_idx,
        col_mx_idx=columns.astype(np.int32),
    )


def _parse_basis_indices(path: Path, n_basis: int) -> AimspyStructure:
    basis_atom: list[int] = []
    basis_l: list[int] = []
    basis_m: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        basis_atom.append(int(fields[2]) - 1)
        basis_l.append(int(fields[4]))
        basis_m.append(int(fields[5]))
    if len(basis_atom) != n_basis:
        raise ValueError(
            f"{path}: parsed {len(basis_atom)} basis rows; expected {n_basis}"
        )
    atom_array = np.asarray(basis_atom, dtype=np.int32)
    n_atoms = int(atom_array.max()) + 1
    return AimspyStructure(
        n_atoms=n_atoms,
        n_basis=n_basis,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3, dtype=np.float64),
        atom_symbols=["X"] * n_atoms,
        atom_coords=np.zeros((n_atoms, 3), dtype=np.float64),
        basis_atom=atom_array,
        basis_l=np.asarray(basis_l, dtype=np.int32),
        basis_m=np.asarray(basis_m, dtype=np.int32),
    )


def load_real_case(directory: Path) -> BenchmarkCase:
    directory = directory.resolve()
    required = {
        "indices": directory / "rs_indices.out",
        "basis": directory / "basis-indices.out",
        "matrix": directory / "rs_hamiltonian.out",
    }
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {', '.join(missing)}")
    descriptor = _parse_rs_indices(required["indices"])
    structure = _parse_basis_indices(required["basis"], descriptor.n_basis)
    entries = np.loadtxt(required["matrix"], dtype=np.float64)
    entries = np.asarray(entries, dtype=np.float64).reshape(-1)
    if entries.size != descriptor.n_ham_size:
        raise ValueError(
            f"{required['matrix']}: contains {entries.size} values; "
            f"expected {descriptor.n_ham_size}"
        )
    return BenchmarkCase(
        "real", entries.reshape(1, -1), descriptor, structure, str(directory), None
    )


def _peak_rss_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _summary(samples: list[float], n_entries: int) -> dict[str, Any]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "samples_seconds": samples,
        "min_seconds": min(samples),
        "median_seconds": median,
        "mad_seconds": mad,
        "throughput_entries_per_second": n_entries / median,
    }


def _profile_case(case: BenchmarkCase) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    blocks = AimspyMatrix.from_aims_csr(case.matrix, case.descriptor, case.structure)
    blocks.to_aims_csr(case.descriptor, case.structure)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    return stream.getvalue()


def run_case(
    case: BenchmarkCase, *, warmups: int, samples: int, profile: bool
) -> dict[str, Any]:
    if warmups < 0 or samples <= 0:
        raise ValueError("warmups must be >= 0 and samples must be > 0")
    for _ in range(warmups):
        matrix = AimspyMatrix.from_aims_csr(
            case.matrix, case.descriptor, case.structure
        )
        matrix.to_aims_csr(case.descriptor, case.structure)

    from_samples: list[float] = []
    to_samples: list[float] = []
    converted: AimspyMatrix | None = None
    roundtrip: np.ndarray | None = None
    for _ in range(samples):
        start = time.perf_counter()
        converted = AimspyMatrix.from_aims_csr(
            case.matrix, case.descriptor, case.structure
        )
        from_samples.append(time.perf_counter() - start)
        start = time.perf_counter()
        roundtrip = converted.to_aims_csr(case.descriptor, case.structure)
        to_samples.append(time.perf_counter() - start)

    assert converted is not None and roundtrip is not None
    n_entries = case.descriptor.n_ham_size - 1
    max_error = float(
        np.max(np.abs(roundtrip[0, :n_entries] - case.matrix[0, :n_entries]))
    )
    digest = hashlib.sha256(case.matrix.tobytes()).hexdigest()
    result: dict[str, Any] = {
        "name": case.name,
        "source": case.source,
        "seed": case.seed,
        "n_basis": case.descriptor.n_basis,
        "n_cells": case.descriptor.n_cells,
        "n_entries": n_entries,
        "block_count": converted.n_pairs,
        "roundtrip_max_error": max_error,
        "peak_rss_mib": _peak_rss_mib(),
        "input_sha256": digest,
        "csr_to_blocks": _summary(from_samples, n_entries),
        "blocks_to_csr": _summary(to_samples, n_entries),
    }
    if case.source == "synthetic":
        synthetic = PROFILES[case.name]
        result["case_parameters"] = {
            "n_basis": synthetic.n_basis,
            "active_cells": synthetic.active_cells,
            "nonzeros_per_row": synthetic.nonzeros_per_row,
            "orbitals_per_atom": synthetic.orbitals_per_atom,
        }
    else:
        result["case_parameters"] = {"directory": case.source}
    if profile:
        result["profile_summary"] = _profile_case(case)
    return result


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor().strip() or platform.machine() or "unknown"


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aimspy": aimspy.__version__,
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "git_revision": _git_revision(),
    }


def _worker_payload(args: argparse.Namespace) -> dict[str, Any]:
    case = (
        load_real_case(args.real_dir)
        if args.real_dir is not None
        else _synthetic_case(args.tier[0], args.seed)
    )
    return {
        "metadata": _environment_metadata(),
        "parameters": {
            "warmups": args.warmups,
            "samples": args.samples,
            "seed": args.seed,
            "profile": args.profile,
        },
        "result": run_case(
            case, warmups=args.warmups, samples=args.samples, profile=args.profile
        ),
    }


def _run_worker(args: argparse.Namespace, tier: str | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--seed",
        str(args.seed),
    ]
    if args.profile:
        command.append("--profile")
    if args.real_dir is not None:
        command.extend(("--real-dir", str(args.real_dir)))
    else:
        assert tier is not None
        command.extend(("--tier", tier))
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        action="append",
        choices=tuple(PROFILES),
        help="synthetic tier; repeat to select several (default: small/medium/large)",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        help="directory containing rs_indices.out, basis-indices.out, and rs_hamiltonian.out",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--json", type=Path, dest="json_path")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.real_dir is not None and args.tier:
        raise SystemExit("--real-dir and --tier cannot be used together")
    if args.real_dir is not None:
        # Child workers run from REPOSITORY_ROOT, so preserve the caller's
        # relative-path meaning before starting the subprocess.
        args.real_dir = args.real_dir.resolve()
    if args.warmups < 0 or args.samples <= 0:
        raise SystemExit("--warmups must be >= 0 and --samples must be > 0")
    if args.worker:
        if args.real_dir is None and (not args.tier or len(args.tier) != 1):
            raise SystemExit("worker mode requires exactly one --tier or --real-dir")
        print(json.dumps(_worker_payload(args), sort_keys=True))
        return 0

    tiers = [] if args.real_dir is not None else args.tier or list(DEFAULT_TIERS)
    workers = (
        [_run_worker(args, None)]
        if args.real_dir is not None
        else [_run_worker(args, tier) for tier in tiers]
    )
    payload = {
        "schema_version": 1,
        "metadata": workers[0]["metadata"],
        "parameters": {
            **workers[0]["parameters"],
            "tiers": tiers,
            "real_dir": str(args.real_dir.resolve()) if args.real_dir else None,
        },
        "results": [worker["result"] for worker in workers],
    }
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_path is None:
        print(output)
    else:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(output + os.linesep, encoding="utf-8")
    if args.profile:
        for result in payload["results"]:
            print(f"\n[{result['name']}] cProfile hotspots", file=sys.stderr)
            print(result["profile_summary"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
