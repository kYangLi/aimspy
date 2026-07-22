AimsPy
======

.. div:: sd-text-left sd-font-italic

    *In-memory Python interface to FHI-aims via ctypes, for seamless integration with DeepX/DeepH-pack*


----

`AimsPy <https://github.com/kYangLi/aimspy>`_ drives `FHI-aims <https://aims-code.rg.mpg.de/>`_ DFT calculations directly from Python — no subprocess, no file-staged I/O on hot paths — by loading a patched ``libaims.so`` via ``ctypes`` and exchanging matrices in memory through a callback framework. It is designed as the FHI-aims binding layer of the `DeepH <https://github.com/kYangLi/DeepH-pack-docs>`_ ecosystem, and the central enabler of **warmstart SCF**: injecting an externally-predicted Hamiltonian (e.g. from a DeepH-trained model) as the initial guess so that a single SCF iteration reproduces the converged result.

AimsPy also establishes a uniform in-memory representation of block-sparse real-space matrices — ``AimspyMatrix`` — that round-trips between FHI-aims' internal CSR layout and the DeepH on-disk format with documented sign/parity conventions, making it equally useful as a standalone post-processing interface for FHI-aims users.

Features
^^^^^^^^

.. grid::

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: In-Memory SCF
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Load ``libaims.so`` once and drive the full SCF cycle from Python via ``ctypes``. No subprocess, no file-staged I/O on the hot path — Hamiltonian, overlap, energy, and forces are exchanged as in-memory arrays through a callback framework.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Warmstart
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Inject an external Hamiltonian (e.g. a DeepH prediction) as the initial guess and converge SCF in a single iteration. Four strategies — ``REPLACE``, ``ADD``, ``SCALE``, ``CUSTOM`` — cover warmstart, perturbation, scaling, and arbitrary user transforms.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Pluggable Matrix Sources
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                The ``ExternalMatrixSource`` protocol accepts any object with ``to_aimspy(structure) -> AimspyMatrix``. A reference ``DeepHData`` adapter ships built-in; adding a new format is a single subpackage under ``aimspy/interface/``.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Bundled FHI-aims Patch
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                ``aimspy patch`` applies, uninstalls, and lists versioned diffs against an FHI-aims source tree — no manual editing. The patch exposes five Fortran callback hook points and the warmstart short-circuit inside ``initialize_scf.f90``.

Installation
^^^^^^^^^^^^

Install the latest release from PyPI:

.. code-block:: bash

    pip install aimspy

AimsPy loads a *patched* ``libaims.so``. To patch an FHI-aims source tree:

.. code-block:: bash

    cd /path/to/FHI-aims
    aimspy patch                 # applies the latest bundled diff

For detailed guidance including uv setup, patch variants, environment variables, and development installation, please refer to `Installation & Setup <./installation_and_setup.html>`_.


Basic usage
^^^^^^^^^^^

AimsPy is primarily a Python API. The most common entry point is the one-shot ``Calculator.do()``:

.. code-block:: python

    from mpi4py import MPI
    from aimspy import Calculator, CalculatorConfig

    config = CalculatorConfig(lib_path="/path/to/libaims.so")
    with Calculator(config) as calc:
        calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
        H = calc.hamiltonian     # AimspyMatrix (block-sparse, Hartree)
        E = calc.energy          # float (Hartree)

Run with MPI:

.. code-block:: bash

    mpiexec -np 8 python script.py

The ``aimspy patch`` command-line tool manages the bundled FHI-aims patch:

.. code-block:: bash

    aimspy patch --help
    aimspy patch --list
    aimspy patch --check /path/to/FHI-aims

For complete examples — baseline SCF, DeepH warmstart, DeepH export, and error recovery — see `Basic Usage <./basic_usage.html>`_.


Citation
^^^^^^^^

If you use this code in your academic work, please cite **the complete package featuring the latest implementation, methodology, and workflow of `DeepH <https://github.com/kYangLi/DeepH-pack-docs>`_**:

`Yang Li, Yanzhen Wang, Boheng Zhao, et al. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026) <https://arxiv.org/abs/2601.02938>`_

.. code-block:: bibtex

    @article{li2026deeph,
        title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
        author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
        journal={arXiv preprint arXiv:2601.02938},
        year={2026}
    }


----

.. toctree::
    :hidden:
    :maxdepth: 1

    installation_and_setup
    basic_usage
    key_concepts
    for_developers/index
    citation_and_license
