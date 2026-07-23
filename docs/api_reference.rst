API Reference
=============

This page documents the full public API of AimsPy. All symbols below
are importable from the top-level ``aimspy`` package
(e.g. ``from aimspy import Calculator``).

Calculator
----------

The main user-facing class for driving FHI-aims SCF calculations,
including lifecycle management (init / calc / do / close / force_close)
and Hamiltonian modification (``modify_init_ham``).

.. automodule:: aimspy.calculator
   :members:
   :show-inheritance:
   :exclude-members: CalculatorConfig

Configuration
-------------

Configuration dataclass for ``Calculator`` — declares lib path, input
files, capture flags, and other construction-time settings.

.. autoclass:: aimspy.calculator.CalculatorConfig
   :members:
   :show-inheritance:

Matrices
--------

Block-sparse real-space matrix representation (``AimspyMatrix``) and
CSR conversion utilities for round-tripping with FHI-aims' internal
layout.

.. automodule:: aimspy.matrix
   :members:
   :show-inheritance:

Structure
---------

Structure and orbital descriptor, providing atom/basis info and derived
properties (phase factor, orbital counts, atom permutation) needed for
matrix conversions.

.. automodule:: aimspy.structure
   :members:
   :show-inheritance:

Runtime Info
------------

Snapshot of FHI-aims runtime dimensions, basis info, and unit
conversion constants (Hartree↔eV, Bohr↔Å).

.. automodule:: aimspy.data
   :members:
   :show-inheritance:
   :exclude-members: CsrMatrixDescriptor

Descriptors
-----------

CSR sparse-storage layout descriptor (``CsrMatrixDescriptor``) —
captures the FHI-aims internal matrix layout needed for
aims↔aimspy conversion.

.. autoclass:: aimspy.data.CsrMatrixDescriptor
   :members:
   :show-inheritance:

Info Loader
-----------

Utility for loading runtime info from the live aims binding.

.. automodule:: aimspy.info
   :members:
   :show-inheritance:

External Matrix Sources
-----------------------

Protocol for pluggable matrix sources used in warmstart — any object
with a ``to_aimspy(structure)`` method satisfies this protocol.

.. automodule:: aimspy.interface
   :members:
   :show-inheritance:

DeepH Data
----------

DeepH on-disk format reader, writer, and converter — reads
``POSCAR`` + ``info.json`` + ``.h5`` files and converts to
``AimspyMatrix``.

.. automodule:: aimspy.interface.deeph
   :members:
   :show-inheritance:

Exceptions
----------

AimsPy-specific exception hierarchy.

.. automodule:: aimspy._exceptions
   :members:
   :show-inheritance:

Callback Identifiers
--------------------

Enum identifying callback types for ``register_callback``.

.. automodule:: aimspy._callbacks.registry
   :members:
   :show-inheritance:
