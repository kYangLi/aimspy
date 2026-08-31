#!/usr/bin/env python
"""MPI integration test for final energy, force, and analytical stress.

The source fixture is copied to an isolated directory below ``/tmp``.  The
script intentionally runs two Calculators in one process to detect stale
observable pointers across FHI-aims lifecycles.

Usage::

    AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so \
        mpiexec -np 4 python tests/test_observables.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimspy import HARTREE_TO_EV, Calculator, CalculatorConfig, DeepHData
from tests.mpi_utils import synchronized_python_exceptions

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2"
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

comm = MPI.COMM_WORLD
rank = comm.rank


def _prepare_case(
    root: Path,
    name: str,
    *,
    forces: bool,
    stress: bool,
    first_atom_shift: float = 0.0,
) -> Path:
    case = root / name
    with synchronized_python_exceptions(comm):
        if rank == 0:
            case.mkdir(parents=True)
            shutil.copy2(DATA_DIR / "geometry.in", case / "geometry.in")
            if first_atom_shift:
                geometry_lines = (case / "geometry.in").read_text().splitlines()
                for index, line in enumerate(geometry_lines):
                    fields = line.split()
                    if fields and fields[0] == "atom":
                        fields[1] = str(float(fields[1]) + first_atom_shift)
                        geometry_lines[index] = " ".join(fields)
                        break
                else:
                    raise AssertionError("fixture has no Cartesian atom line")
                (case / "geometry.in").write_text("\n".join(geometry_lines) + "\n")
            lines = []
            for line in (DATA_DIR / "control.in").read_text().splitlines():
                key = line.strip().split(maxsplit=1)
                if key and key[0] in {
                    "compute_forces",
                    "compute_analytical_stress",
                }:
                    continue
                lines.append(line)
            if forces:
                lines.append("compute_forces .true.")
            if stress:
                lines.append("compute_analytical_stress .true.")
            (case / "control.in").write_text("\n".join(lines) + "\n")
    comm.Barrier()
    return case


def _run(case: Path, logfile: str, export_dir: Path | None = None):
    calc = Calculator(
        CalculatorConfig(lib_path=lib_path, logfile=Path(logfile), log_level="INFO")
    )
    try:
        calc.do(comm=comm, work_dir=case)
        values = (
            calc.energy,
            calc.energy_raw,
            calc.energy_free,
            calc.free_atom_reference_energies,
            calc.free_atom_reference_energy,
            calc.energy_free_relative,
            None if calc.forces is None else calc.forces.copy(),
            None if calc.stress is None else calc.stress.copy(),
        )
        with synchronized_python_exceptions(comm):
            if export_dir is not None and rank == 0:
                data = DeepHData.from_aimspy(
                    calc.structure,
                    hamiltonian=calc.hamiltonian,
                    force=calc.forces,
                    energy=calc.energy_free_relative,
                    stress=calc.stress,
                )
                data.save(export_dir)
                loaded = DeepHData.from_directory(export_dir)
                if data.force is None:
                    assert loaded.force is None
                else:
                    np.testing.assert_allclose(
                        loaded.force, data.force, rtol=0.0, atol=0.0
                    )
                if data.stress is None:
                    np.testing.assert_array_equal(loaded.stress, np.zeros((3, 3)))
                else:
                    np.testing.assert_allclose(
                        loaded.stress, data.stress, rtol=0.0, atol=0.0
                    )
                assert loaded.energy_eV == data.energy_eV
    finally:
        calc.close()
    assert calc.forces is None
    assert calc.stress is None
    comm.Barrier()
    return values


def _last_energy_ev(text: str, label: str) -> float:
    values = re.findall(rf"\| {re.escape(label)}\s*:\s*({FLOAT})\s+eV", text)
    if not values:
        raise AssertionError(f"energy label not found in output: {label}")
    return float(values[-1])


def _final_stress(text: str) -> np.ndarray:
    marker = "Analytical stress tensor - Symmetrized"
    if marker not in text:
        raise AssertionError("final analytical stress section not found")
    section = text.rsplit(marker, maxsplit=1)[1]
    rows = []
    for axis in "xyz":
        match = re.search(
            rf"\|\s+{axis}\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+\|",
            section,
        )
        if match is None:
            raise AssertionError(f"analytical stress row not found: {axis}")
        rows.append([float(value) for value in match.groups()])
    return np.asarray(rows)


lib_env = os.environ.get("AIMSPY_TEST_AIMS_LIBPATH")
if not lib_env:
    if rank == 0:
        print("AIMSPY_TEST_AIMS_LIBPATH is required", file=sys.stderr)
    comm.Abort(1)
lib_path = Path(lib_env)

tmp_name = None
if rank == 0:
    tmp_name = tempfile.mkdtemp(prefix="aimspy-observables-", dir="/tmp")
tmp_root = Path(comm.bcast(tmp_name, root=0))

with_observables = _prepare_case(tmp_root, "with_observables", forces=True, stress=True)
export_dir = tmp_root / "deeph_export"
legacy, raw, free, references, reference_sum, relative, forces, stress = _run(
    with_observables, "observables.out", export_dir
)

assert np.isfinite([legacy, raw, free, reference_sum, relative]).all()
assert references.shape == (2,) and np.isfinite(references).all()
np.testing.assert_allclose(relative, free - reference_sum, rtol=0.0, atol=1e-12)
np.testing.assert_allclose(legacy, raw, rtol=0.0, atol=1e-12)
assert forces is not None and forces.shape == (3, 3) and np.isfinite(forces).all()
assert stress is not None and stress.shape == (3, 3) and np.isfinite(stress).all()

for other in comm.allgather(
    (legacy, raw, free, references, reference_sum, relative, forces, stress)
):
    np.testing.assert_allclose(other[0:3], (legacy, raw, free), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(other[3], references, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        other[4:6], (reference_sum, relative), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(other[6], forces, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(other[7], stress, rtol=0.0, atol=1e-10)

output = (with_observables / "observables.out").read_text()
np.testing.assert_allclose(
    raw * HARTREE_TO_EV,
    _last_energy_ev(output, "Total energy uncorrected"),
    rtol=0.0,
    atol=5e-8,
)
np.testing.assert_allclose(
    free * HARTREE_TO_EV,
    _last_energy_ev(output, "Electronic free energy"),
    rtol=0.0,
    atol=5e-8,
)
np.testing.assert_allclose(stress, _final_stress(output), rtol=0.0, atol=5e-8)

with synchronized_python_exceptions(comm):
    if rank == 0:
        with h5py.File(export_dir / "force.h5", "r") as h5:
            assert set(h5.keys()) == {"cell", "energy", "force", "stress"}
            np.testing.assert_allclose(
                h5["energy"][()], relative * HARTREE_TO_EV, rtol=0.0, atol=1e-10
            )
            np.testing.assert_allclose(
                h5["stress"][:],
                [
                    stress[0, 0],
                    stress[1, 1],
                    stress[2, 2],
                    stress[1, 2],
                    stress[0, 2],
                    stress[0, 1],
                ],
                rtol=0.0,
                atol=1e-12,
            )

without_observables = _prepare_case(
    tmp_root, "without_observables", forces=False, stress=False
)
(
    _,
    raw_without,
    free_without,
    references_without,
    reference_sum_without,
    relative_without,
    forces_without,
    stress_without,
) = _run(
    without_observables,
    "no_observables.out",
    tmp_root / "deeph_without_observables",
)
assert np.isfinite([raw_without, free_without]).all()
np.testing.assert_allclose(references_without, references, rtol=0.0, atol=1e-12)
np.testing.assert_allclose(reference_sum_without, reference_sum, rtol=0.0, atol=1e-12)
np.testing.assert_allclose(relative_without, free_without - reference_sum_without)
assert forces_without is None
assert stress_without is None

shifted_geometry = _prepare_case(
    tmp_root,
    "shifted_geometry",
    forces=False,
    stress=False,
    first_atom_shift=0.137,
)
(
    _,
    _,
    _,
    shifted_references,
    _,
    _,
    _,
    _,
) = _run(shifted_geometry, "shifted_geometry.out")
np.testing.assert_allclose(shifted_references, references, rtol=0.0, atol=1e-12)
for other in comm.allgather(shifted_references):
    np.testing.assert_allclose(other, references, rtol=0.0, atol=1e-12)

with synchronized_python_exceptions(comm):
    if rank == 0:
        with h5py.File(tmp_root / "deeph_without_observables" / "force.h5", "r") as h5:
            assert set(h5.keys()) == {"cell", "energy", "stress"}
            np.testing.assert_array_equal(h5["stress"][:], np.zeros(6))

if rank == 0:
    print(
        "OBSERVABLES TEST PASSED: "
        f"references={references.tolist()} "
        f"reference_sum={reference_sum:.16g} Ha "
        f"relative_free_energy={relative:.16g} Ha "
        f"artifacts={tmp_root}"
    )
