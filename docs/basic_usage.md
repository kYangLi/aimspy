# Basic Usage

AimsPy is primarily used through its Python API, with a small command-line interface for managing the bundled FHI-aims patch. This page walks through the common workflows.

> **Note**: Forward SCF calculations work with any system type. However,
> extracting or injecting matrices (Hamiltonian, overlap, H_init — used
> in warmstart, capture, and export workflows below) requires a
> **periodic system** with `use_local_index = .false.`. For isolated
> molecules, use a sufficiently large periodic cell with vacuum (see the
> `from_scratch/run.py` example).

## 1. Python API

### Baseline SCF

The most common entry point is the one-shot `Calculator.do()` — `init()` + `calc()` — wrapped in a context manager:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig

comm = MPI.COMM_WORLD
rank = comm.rank

config = CalculatorConfig(lib_path="/path/to/libaims.so")
with Calculator(config) as calc:
    calc.do(comm=comm, work_dir="./MoS2")
    if rank == 0:
        H = calc.hamiltonian     # AimspyMatrix (block-sparse, Hartree, rank-0 only)
        S = calc.overlap         # rank-0 fallback (see capture_overlap below)
    E = calc.energy          # float (Hartree)
    F = calc.forces          # (n_atoms, 3) ndarray in eV/Å, or None
```

Run with MPI:

```bash
mpiexec -np 8 python script.py
```

`work_dir` must contain `control.in` + `geometry.in` (AimsPy `chdir`s into it because FHI-aims uses `chdir(2)` internally — this also makes the `Calculator` **not thread-safe**).

### Export to DeepH format

Export converged Hamiltonian, overlap, and free-atom initial Hamiltonian to the DeepH on-disk format — ideal for generating training data:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig
from aimspy import DeepHData

comm = MPI.COMM_WORLD
rank = comm.rank

config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,  # capture free-atom H0
)
with Calculator(config) as calc:
    calc.do(comm=comm, work_dir="./MoS2")

    # Export H, S, H0 to DeepH on-disk format
    if rank == 0:
        dd = DeepHData.from_aimspy(
            calc.structure,
            hamiltonian=calc.hamiltonian,
            overlap=calc.overlap,
            initial_hamiltonian=calc.initial_hamiltonian,
        )
        dd.save("deeph_out/")
```

> **Note**: For the DeepH on-disk data format specification (POSCAR, info.json,
> .h5 files), see [DeepH-dock Key Concepts](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html).

### DeepH warmstart

The central use case for AimsPy. Inject a pre-trained DeepH Hamiltonian as the initial guess and converge SCF in several iterations:

```python
from mpi4py import MPI
from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy import DeepHData

data = DeepHData.from_directory("deeph_out/")
config = CalculatorConfig(lib_path="/path/to/libaims.so")
calc = Calculator(config)
calc.modify_init_ham(source=data, strategy=Strategy.REPLACE)
calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
```

**Deferred source** — generate the source at runtime (after H0/overlap are available, inside the `python_func` callback):

```python
config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,
)
calc = Calculator(config)

@calc.modify_init_ham(strategy=Strategy.REPLACE, option={"deeph_path": "deeph_out/"})
def gen_source(calculator, option):
    # calculator.initial_hamiltonian / .overlap available here
    return DeepHData.from_directory(option["deeph_path"])

calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
```

### Modification strategies

The `Strategy` enum covers the common H0-modification cases:

| Strategy | Behaviour | Required argument |
|----------|-----------|-------------------|
| `REPLACE` | Clear and copy the external Hamiltonian's blocks into the live H0 buffer | `source=` |
| `ADD`     | Add external blocks on top of the live H0 (e.g. a predicted H − H₀ to recover H) | `source=` |
| `SCALE`   | Multiply the live H0 by a constant factor | `factor=` (float) |
| `CUSTOM`  | Call `custom_fn(live, external, structure, aux)` to mutate the live matrix in place | `custom_fn=` (callable) |

For technical details on each strategy and the `modify_init_ham` API (direct vs. deferred mode, state guards), see [Key Concepts](./key_concepts.md#hamiltonian-modification-strategies).

### Capturing overlap and the free-atom H0

Two `CalculatorConfig` flags opt in to additional callbacks:

```python
config = CalculatorConfig(
    lib_path="/path/to/libaims.so",
    capture_initial_hamiltonian=True,   # export_h0 callback
    capture_overlap=True,               # export_ovlp callback (live, all ranks)
)
with Calculator(config) as calc:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
    H0 = calc.initial_hamiltonian        # free-atom H_init (AimspyMatrix)
    S  = calc.overlap                    # live overlap on all ranks
```

Without `capture_overlap`, `calc.overlap` falls back to a rank-0 snapshot taken after `calc()`.

### Error recovery

If SCF crashes, use `force_close()` (always safe — swallows Fortran errors from any state) and create a fresh `Calculator`. FHI-aims is a **global singleton**: one `init`/`finalize` cycle per process, so a finalized `Calculator` cannot be reused.

**Context manager** (recommended — `__exit__` auto-calls `force_close()` on exception):

```python
config = CalculatorConfig(lib_path="...")
try:
    with Calculator(config) as calc:
        calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
        H = calc.hamiltonian
except Exception as e:
    print(f"SCF failed: {e}")
    # Calculator already force_closed by __exit__; create a new one to retry
```

**Manual pattern** (when you need finer control):

```python
calc = Calculator(CalculatorConfig(lib_path="..."))
try:
    calc.do(comm=MPI.COMM_WORLD, work_dir="./bad_input")
except Exception:
    calc.force_close()
    # create a new Calculator for the next run
```

`close()` is the graceful counterpart — silent no-op from `UNINIT`/`FINALIZED`, raises `AimspyStateError` from `RUNNING` (use `force_close`), and a normal finalize from `INITED`/`DONE`.

## 2. Command-line Tool

The `aimspy patch` CLI manages the bundled FHI-aims patch (apply, uninstall, dry-run, list). See the [CLI reference](./cli.md) for full options and examples.

## 3. Learning Through Examples

The [`examples/`](https://github.com/kYangLi/aimspy/tree/main/examples) directory in the repository contains two runnable end-to-end scripts:

- **`from_scratch/run.py`** — H₂O baseline SCF + DeepH export. Demonstrates `CalculatorConfig.capture_initial_hamiltonian=True` + `capture_overlap=True`, and `DeepHData.from_aimspy(...).save(...)`.
- **`continue_calc/run.py`** — warmstart demo. Loads the previous run's DeepH output via `DeepHData.from_directory`, applies `Strategy.REPLACE` via `modify_init_ham(source=data)`, and shows SCF converging in several iterations.

Run them with:

```bash
export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
make run-from-scratch
make run-continue-calc    # requires run-from-scratch first
```

A MoS₂ integration test fixture (including a reference `rs_hamiltonian.out`) is available under `tests/data/MoS2/` for cross-validation.

## 4. Extending AimsPy

AimsPy is designed with extensibility in mind. If you want to add new functionality:

1. **New external matrix source** — implement the `ExternalMatrixSource` protocol in a new subpackage under `aimspy/interface/<your_format>/`.
2. **New callback** — follow the extension contract documented in the [Development Guide](./for_developers/development_guide.md#adding-a-new-callback).
3. **New modification strategy** — extend the `Strategy` enum and the `_apply_strategy` dispatcher.

For detailed guidance, refer to the [Development Guide](./for_developers/development_guide.md).

## Need Help?

- Use `aimspy patch --help` for CLI assistance.
- Check the [`examples/`](https://github.com/kYangLi/aimspy/tree/main/examples) directory for practical implementations.
- For technical background on the in-memory architecture, callbacks, and data formats, see [Key Concepts](./key_concepts.md).
- For development questions, see the [For Developers](./for_developers/index.rst) section.
