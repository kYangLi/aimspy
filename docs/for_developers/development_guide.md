# Development Guide

This guide details the technical foundations of AimsPy, explaining its architecture, data formats, and how to extend it with new functionality. It is intended for developers who wish to integrate new matrix sources, add new callbacks, or contribute to the core package.

## Architecture Overview

AimsPy is built around a layered architecture. All user-facing code lives in the top layer; the lower layers are private and explicitly marked as such.

| Layer | Module | Purpose |
|-------|--------|---------|
| Public API | `aimspy.calculator`, `aimspy.matrix`, `aimspy.structure`, `aimspy.data`, `aimspy.info`, `aimspy.exceptions` | User-facing classes and exceptions |
| Interface adapters | `aimspy.interface`, `aimspy.interface.deeph` | `ExternalMatrixSource` protocol + `DeepHData` reference adapter |
| Callback framework | `aimspy._callbacks` | `CallbackManager`, `CallbackSpec`, `CallbackName` enum |
| ctypes binding | `aimspy._binding` | `BindingLib`, `libloader`, Fortran struct mirrors, `CFUNCTYPE` types |
| FHI-aims patch | `aimspy._patches`, `aimspy.cli` | Versioned diffs + `aimspy patch` CLI |

The data flows top-down at construction time (config → binding → callbacks) and bottom-up at runtime (Fortran callbacks → Python → `AimspyMatrix` → user).

### Single sources of truth

AimsPy is deliberately conservative about where state lives:

- `_runtime_aux` (a `dict` on `Calculator`) — callback scratch state, populated by `_wire_callbacks` and read by the `modify_h0` wrapper.
- `_modify` (a `SimpleNamespace` on `Calculator`) — H0 modification config (`source`, `strategy`, `factor`, `custom_fn`, `option`).
- `CALLBACK_SPECS` (in `aimspy._callbacks.registry`) — the authoritative catalogue of all callback types.

## Data Formats & Specifications

AimsPy's canonical in-memory representation is the **`AimspyMatrix` block-sparse format**, whose conventions (sign, parity, units, Hermitian partners) are documented in [Key Concepts](../key_concepts.md#aimspymatrix-block-sparse-format). All developers should read that section before touching `matrix.py` or `interface/deeph/`.

The DeepH on-disk format is shared with [DeepH-dock](https://github.com/kYangLi/DeepH-dock); its full field-level specification is in the [DeepH-dock Key Concepts](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html) page.

## Extending AimsPy

After forking the source code by following the [Fork and Pull Request Process](./collaboration_guide.md#fork-and-pull-request-process), you can set up your development environment by referring to [Install from Source (for Developers)](../installation_and_setup.md#install-from-source-for-developers).

### Adding a New Callback

Adding a new callback type requires touching the following well-defined places (documented at the top of `aimspy/_callbacks/base.py`):

1. **Fortran patch** (`aimspy/_patches/aimspy-patch_vX.Y.Z.diff`) — add the abstract callback interface in `callback.f90`, the `aimspy_register_<name>_callback` subroutine in `register.f90`, and the trigger point in `initialize_scf.f90`. Bump the patch version and update the `Makefile` `PATCH_VERSION` line.

2. **`aimspy/_binding/callback_types.py`** — add a `CFUNCTYPE` declaration matching the Fortran abstract interface:

   ```python
   MyNewCb = CFUNCTYPE(None, c_void_p, c_void_p)  # (aux, extra_ptr) — adjust to taste
   ```

3. **`aimspy/_binding/prototypes.py`** — add an entry to `_PROTOTYPES` mapping the register symbol to its `(argtypes, restype)`. `setup_prototypes` will pick it up automatically and `BindingLib.has(name)` will probe availability at runtime.

4. **`aimspy/_callbacks/registry.py`** — add a `CallbackSpec` entry to `CALLBACK_SPECS`:

   ```python
   CallbackSpec(
       name="my_new_cb",
       ctypes_type=MyNewCb,
       register_symbol="aimspy_register_my_new_cb_callback",
       register_arg_count=2,         # 3 if the Fortran register takes an extra c_ptr
       trigger_stage="pre_scf",       # or "post_scf" if you add a new stage
       fortran_module="initialize_scf.f90:NNN",
   ),
   ```

5. **`aimspy/_callbacks/base.py`** — add a branch to `_build_ctypes_wrapper` that unpacks the `aux` and converts the C pointer arguments to numpy views before calling the user function. Follow the existing `export_ovlp` / `export_h0` branches (and mark `intent(in)` views `writeable=False`).

That's it — `CallbackManager.register`, `Calculator.register_callback`, and `CallbackName` all derive from `CALLBACK_SPECS`, so the new callback is automatically wired through.

### Adding a New External Matrix Source

To plug in a new on-disk format (e.g. a different DFT code's output):

1. **Create a subpackage** under `aimspy/interface/<your_format>/` (e.g. `aimspy/interface/openmx/`).

2. **Implement a class** satisfying the `ExternalMatrixSource` protocol:

   ```python
   from aimspy.matrix import AimspyMatrix
   from aimspy.structure import AimspyStructure

   class OpenMXData:
       """Reads OpenMX output and converts to AimspyMatrix."""

       @classmethod
       def from_directory(cls, path: str) -> "OpenMXData":
           ...

       def to_aimspy(self, structure: AimspyStructure) -> AimspyMatrix:
           # Build the block-sparse dict following the conventions in
           # key_concepts.md#aimspymatrix-block-sparse-format:
           #   - R_aimspy = -R_aims = R_deeph
           #   - aims native atom/orbital order
           #   - phase_i * phase_j already applied
           #   - Hartree units
           #   - both (R,i,j) and (-R,j,i) stored
           ...
   ```

3. **Use it** — `Calculator.modify_init_ham(source=OpenMXData.from_directory("..."), strategy=Strategy.REPLACE)` will work with no other changes.

The `ExternalMatrixSource` protocol is `@runtime_checkable`, so `isinstance(obj, ExternalMatrixSource)` works for validation. The deferred-source decorator path also works: any object with `to_aimspy(structure) -> AimspyMatrix` is accepted.

### Adding a New Modification Strategy

Extend the `Strategy` enum in `aimspy/calculator.py` and add a branch to `_apply_strategy`. The strategy receives `(live, external, structure, aux)` and must mutate `live` (an `AimspyMatrix`) in place.

## Testing

AimsPy separates unit tests (no MPI, no `libaims`) from integration tests (require `AIMSPY_TEST_AIMS_LIBPATH` + `mpiexec`):

```bash
make test                 # unit tests only (pytest -v)
make test-integration     # 6 MPI integration tests, in dependency order
make test-all             # both
```

### Unit tests — `tests/unit/`

No MPI, no `libaims`. Cover `AimspyStructure` derived properties, `DeepHData` I/O roundtrips, POSCAR parsing, `Strategy` / `CallbackName` enums, the `ExternalMatrixSource` protocol, and `force_close` / `CalcState` transitions. `tests/conftest.py` provides `mock_structure` fixtures (3-atom MoS₂-like).

### Integration tests — `tests/test_*.py`

Require `AIMSPY_TEST_AIMS_LIBPATH` pointing at a patched `libaims.so`, run under `mpiexec` (default 8 ranks). The suite covers baseline SCF, DeepH export, warmstart, overlap capture, regression (50+ checks), and all `Strategy` variants.

> **Important**: FHI-aims is a global Fortran singleton — one `init`/`finalize` per process. `test_strategies.py` therefore runs each strategy in a **separate MPI invocation via `subprocess`**, with the controlling Python process dispatching and aggregating results. New per-strategy tests should follow this pattern.

### Test data

`tests/data/MoS2/` contains a MoS₂ fixture (`control.in`, `geometry.in`) and reference outputs (`rs_hamiltonian.out`, `rs_overlap.out`, `rs_indices.out`, `basis-indices.out`). Integration tests cross-validate against these references.

## Best Practices & Conventions

- **Code style**: `ruff check .` and `black --check .` must pass. `make lint` runs both.
- **Caching**: use `@functools.cached_property` for derived properties on immutable dataclasses (see `AimspyStructure.phase_factor`, `basis_subidx`, `atom_permutation`).
- **Logging**: INFO/WARNING emitted on rank 0 only; ERROR on all ranks. AimsPy attaches a `NullHandler` to the `aimspy` logger — it never configures the root logger.
- **GC safety**: ctypes wrappers and Python objects passed to Fortran via `c_void_p.from_buffer(py_object(aux))` must be kept alive explicitly. See `CallbackManager._pyobjs` / `_wrapped` / `_auxs`.
- **State guards**: `Calculator._state_guard` raises `AimspyStateError` on illegal transitions. New methods that touch Fortran should declare their allowed states.
- **No `os` outside `aimspy._system`**: `_system.py` is the only module that imports `os`. All other modules use `pathlib.Path`. This keeps `chdir(2)` usage auditable.
- **Documentation**: include docstrings for all public functions and classes. Update the main documentation under `docs/` if the public API changes.

## Next Steps

After familiarizing yourself with these concepts, you are ready to contribute code. Please follow the collaborative process outlined in the [Collaboration Guide](./collaboration_guide.md) to submit your changes.
