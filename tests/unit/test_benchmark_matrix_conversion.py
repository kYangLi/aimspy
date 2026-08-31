from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.benchmark_matrix_conversion import (
    _synthetic_case,
    load_real_case,
    run_case,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "tests" / "benchmark_matrix_conversion.py"


def _write_real_fixture(directory: Path) -> None:
    case = _synthetic_case("tiny", seed=17)
    descriptor = case.descriptor
    lines = [
        f"n_hamiltonian_matrix_size: {descriptor.n_ham_size}",
        f"n_cells_in_hamiltonian: {descriptor.n_cells}",
        f"n_basis: {descriptor.n_basis}",
        "cell_index",
    ]
    lines.extend(
        " ".join(map(str, descriptor.cell_idx[:, cell]))
        for cell in range(descriptor.n_cells)
    )
    lines.append("index_hamiltonian(1,:,:)")
    lines.extend(
        " ".join(map(str, descriptor.row_mx_idx[:, cell, 0]))
        for cell in range(descriptor.n_cells)
    )
    lines.append("index_hamiltonian(2,:,:)")
    lines.extend(
        " ".join(map(str, descriptor.row_mx_idx[:, cell, 1]))
        for cell in range(descriptor.n_cells)
    )
    lines.append("column_index_hamiltonian")
    lines.extend(str(value) for value in descriptor.col_mx_idx)
    (directory / "rs_indices.out").write_text("\n".join(lines), encoding="utf-8")

    basis_lines = ["fn. type at. n l m"]
    for index, atom in enumerate(case.structure.basis_atom, start=1):
        basis_lines.append(f"{index} atomic {int(atom) + 1} 1 0 0")
    (directory / "basis-indices.out").write_text(
        "\n".join(basis_lines), encoding="utf-8"
    )
    np.savetxt(directory / "rs_hamiltonian.out", case.matrix.reshape(-1))


def test_tiny_synthetic_roundtrip_is_exact():
    result = run_case(_synthetic_case("tiny", 31), warmups=0, samples=1, profile=False)

    assert result["roundtrip_max_error"] == 0.0
    assert result["n_entries"] == 144
    assert result["block_count"] > 0


def test_synthetic_seed_is_reproducible():
    first = run_case(_synthetic_case("tiny", 47), warmups=0, samples=1, profile=False)
    second = run_case(_synthetic_case("tiny", 47), warmups=0, samples=1, profile=False)

    assert first["input_sha256"] == second["input_sha256"]


def test_json_output_schema_and_types(tmp_path):
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tier",
            "tiny",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--json",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert isinstance(payload["metadata"]["python"], str)
    assert isinstance(payload["metadata"]["numpy"], str)
    assert isinstance(payload["metadata"]["aimspy"], str)
    assert isinstance(payload["metadata"]["platform"], str)
    assert isinstance(payload["metadata"]["cpu"], str)
    assert isinstance(payload["metadata"]["git_revision"], str)
    result = payload["results"][0]
    assert isinstance(result["block_count"], int)
    assert isinstance(result["roundtrip_max_error"], float)
    assert result["case_parameters"]["nonzeros_per_row"] == 4
    assert isinstance(result["csr_to_blocks"]["median_seconds"], float)
    assert isinstance(result["blocks_to_csr"]["mad_seconds"], float)


def test_real_format_fixture_is_parsed_and_roundtrips(tmp_path):
    _write_real_fixture(tmp_path)

    case = load_real_case(tmp_path)
    result = run_case(case, warmups=0, samples=1, profile=False)

    assert case.descriptor.n_basis == 12
    assert result["source"] == str(tmp_path.resolve())
    assert result["roundtrip_max_error"] == 0.0


def test_relative_real_directory_keeps_callers_working_directory(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_real_fixture(case_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--real-dir",
            case_dir.name,
            "--warmups",
            "0",
            "--samples",
            "1",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["results"][0]["source"] == str(case_dir.resolve())
    assert payload["results"][0]["roundtrip_max_error"] == 0.0


def test_profile_mode_reports_hotspots():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--tier",
            "tiny",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--profile",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert "function calls" in payload["results"][0]["profile_summary"]
    assert "cProfile hotspots" in completed.stderr
