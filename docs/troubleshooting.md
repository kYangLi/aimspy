# Troubleshooting & FAQ

Common issues and their solutions. If your problem is not listed here,
please [open a GitHub issue](https://github.com/kYangLi/aimspy/issues).

## Installation & Build

### `libaims.so` not found or cannot be loaded

**Cause**: FHI-aims was built as a static library (default), or the path
is incorrect.

**Solution**: Rebuild with `-DBUILD_SHARED_LIBS=ON`:

```bash
cd /path/to/FHI-aims/build
cmake -DUSE_MPI=ON -DUSE_LIBXS=OFF -DBUILD_SHARED_LIBS=ON ..
make -j 8
```

See [Installation & Setup](./installation_and_setup.md#patching-fhi-aims)
for full build instructions. AimsPy loads `libaims.so` via `ctypes`, so a
shared library build is **required**.

### `aimspy patch --check` fails

**Cause**: The FHI-aims source tree is not on the patch's base branch
(currently `dev`), or the tree has been modified.

**Solution**:

```bash
cd /path/to/FHI-aims
git checkout dev
git clean -fd   # remove untracked files
aimspy patch --check
```

The patch must apply on a clean checkout. If you have previously applied
a different patch version, uninstall it first:

```bash
aimspy patch --uninstall /path/to/FHI-aims
```

> **Note**: The current patch supports FHI-aims versions **250822** and
> **250822_1** only. If your FHI-aims checkout is a different version,
> the patch will not apply. Patches for additional versions will be
> released in the future.

### MPI errors or severe performance degradation

**Cause**: `mpi4py` and `libaims.so` are linked against different MPI
libraries (e.g. `mpi4py` uses system OpenMPI while `libaims.so` uses
Intel MPI).

**Solution**: Source the Intel OneAPI environment **before** installing
AimsPy and **before** building `libaims.so`:

```bash
source /path/to/intel/setvars.sh
pip install aimspy          # mpi4py compiles against Intel MPI
```

See [Prerequisites](./installation_and_setup.md#prerequisites-intel-oneapi-environment)
for details.

## Runtime Errors

### `AimspyStateError: Cannot init() in state ...`

**Cause**: The `Calculator` is not in the `UNINIT` state. FHI-aims is a
**global singleton** — only one `init`/`finalize` cycle per process.

**Solution**: Create a **new** `Calculator` instance for each SCF run.
After `close()` or `force_close()`, the current instance is finalized
and cannot be reused:

```python
# First run
calc = Calculator(config)
calc.do(comm, work_dir="./run1")
calc.close()

# Second run — must create a NEW Calculator
calc2 = Calculator(config)
calc2.do(comm, work_dir="./run2")
calc2.close()
```

### `AimspyStateError: Cannot close() in RUNNING state`

**Cause**: `close()` was called while SCF is still running (inside
`calc()`).

**Solution**: Use `force_close()` instead — it is safe from any state
and swallows Fortran errors:

```python
try:
    calc.do(comm, work_dir="./bad_input")
except Exception:
    calc.force_close()
```

### `AimspyBindingError: c_rs_hamiltonian() returned NULL`

**Cause**: A rank-0-only property was accessed on a non-root MPI rank.
The properties `rs_hamiltonian`, `rs_overlap`, `hamiltonian`, and
`overlap` (without `capture_overlap=True`) read from Fortran rank-0
buffers and raise `AimspyBindingError` on non-root ranks.

**Solution**: Guard with `if rank == 0:` or use the all-rank
alternatives:

```python
if rank == 0:
    H = calc.hamiltonian       # rank-0 only

# All-rank alternatives:
S = calc.overlap              # all ranks if capture_overlap=True
h0 = calc.initial_hamiltonian # all ranks if capture_initial_hamiltonian=True
info = calc.info              # all ranks
```

### `AimspyBindingError: C function ... not available in loaded libaims`

**Cause**: The loaded `libaims.so` was built with an older (or
mismatched) patch version that does not export the requested symbol.

**Solution**: Re-patch and rebuild FHI-aims with the latest bundled
patch:

```bash
aimspy patch --uninstall /path/to/FHI-aims   # remove old patch
aimspy patch /path/to/FHI-aims               # apply latest
# Rebuild libaims.so as described in Installation & Setup
```

### `AimspyCallbackError: N callback error(s) during calc()`

**Cause**: One or more Python callbacks raised an exception during SCF.
AimsPy catches callback exceptions (to avoid crashing Fortran) and
aggregates them into a single `AimspyCallbackError` after `aimspy_run`
returns.

**Solution**: Inspect the `callback_errors` attribute for details:

```python
try:
    calc.do(comm, work_dir="./MoS2")
except AimspyCallbackError as e:
    for name, exc, tb_str in e.callback_errors:
        print(f"[{name}] {exc}")
        print(tb_str)
```

### `modify_init_ham` seems to have no effect

**Cause**: `modify_init_ham()` was called **after** `init()` or `do()`.
The callback wiring (`_wire_callbacks`) runs inside `init()`, so setting
`self._modify` afterwards has no effect — the `modify_h0` callback is
never registered.

**Solution**: Always call `modify_init_ham()` **before** `init()` /
`do()`:

```python
calc = Calculator(config)
calc.modify_init_ham(source=data, strategy=Strategy.REPLACE)  # BEFORE do()
calc.do(comm, work_dir="./MoS2")                               # correct order
```

## Common Gotchas

### FHI-aims is a global singleton

Only one `init`/`finalize` cycle per process. A finalized `Calculator`
cannot be reused — always create a new instance. This also means
`Calculator` is **not thread-safe** (FHI-aims uses `chdir(2)` internally).

### `work_dir` must contain `control.in` and `geometry.in`

AimsPy `chdir`s into `work_dir` because FHI-aims uses `chdir(2)`
internally. The directory must contain `control.in` and `geometry.in`
(or use `CalculatorConfig.control_path` / `geometry_path` to have them
copied automatically).

### Forces return `None`

`calc.forces` returns `None` when:
- `calc()` has not been called yet
- `compute_forces .true.` is not set in `control.in`
- The `Calculator` has been closed/finalized

Add `compute_forces .true.` to your `control.in` to enable force
calculation.

### Matrix extraction/injection requires periodic systems

Forward SCF calculations work with any system type. However, extracting
or injecting matrices (Hamiltonian, overlap, H_init — via
`capture_overlap`, `capture_initial_hamiltonian`, `modify_init_ham`, or
reading `calc.hamiltonian` / `calc.overlap`) requires:

1. A **periodic system** (lattice vectors in `geometry.in`, `k_grid` in
   `control.in`)
2. `use_local_index = .false.`

For isolated molecules, use a sufficiently large periodic cell with vacuum
and a Gamma-only `k_grid`:

```
# geometry.in — periodic box with vacuum
lattice_vector  30.0  0.0  0.0
lattice_vector   0.0 30.0  0.0
lattice_vector   0.0  0.0 30.0
```

```ini
# control.in
k_grid 1 1 1
```

The `from_scratch/run.py` example demonstrates this vacuum-box approach
for H₂O.
