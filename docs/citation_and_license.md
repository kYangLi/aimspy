# Citation & License

## Citation

If you use this code in your academic work, please cite **the complete package featuring the latest implementation, methodology, and workflow of [DeepH](https://github.com/kYangLi/DeepH-pack-docs)**:

[Yang Li, Yanzhen Wang, Boheng Zhao, *et al*. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)](https://arxiv.org/abs/2601.02938)

```bibtex
@article{li2026deeph,
    title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
    author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
    journal={arXiv preprint arXiv:2601.02938},
    year={2026}
}
```

## License

AimsPy is released under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

```text
Copyright (C) 2026 DeepH-pack developers

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
```

### What does this mean?

- **Freedom to use**: You can use AimsPy for any purpose
- **Freedom to study**: You have access to the source code
- **Freedom to modify**: You can modify the code to suit your needs
- **Freedom to distribute**: You can share your modifications, but they must also be under GPL-3.0-or-later

For more details, see the full [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.html) or the [LICENSE](https://github.com/kYangLi/aimspy/blob/main/LICENSE) file in the repository.

### FHI-aims is not distributed with AimsPy

AimsPy loads a *patched* `libaims.so` at runtime, but **FHI-aims itself is not distributed with AimsPy** and remains under its own licence agreement with the aims team. Users must obtain the FHI-aims source code independently from the [aims team](https://aims-code.rg.mpg.de/). The bundled patch is an open-source diff against a clean FHI-aims checkout; applying it does not change FHI-aims' licensing for the underlying code.

## Acknowledgements

AimsPy is made possible by the collective efforts of many individuals and organizations:

### Core Development Team

The primary developers and maintainers who have contributed significant time and expertise to build and improve AimsPy.

### Community Contributors

We extend our sincere gratitude to all community members who have:

- Submitted bug reports and feature requests
- Contributed code improvements and new features
- Helped improve documentation and examples
- Shared their use cases and provided valuable feedback

### Supporting Institutions

AimsPy development has been supported by research grants and computing resources from various academic institutions and funding agencies.

### Open Source Community

AimsPy builds upon the work of many open-source projects in the scientific Python ecosystem. We gratefully acknowledge the developers of:

- **NumPy** for numerical computing foundations
- **h5py / HDF5** for efficient data storage
- **mpi4py / MPI** for high-performance parallel computing
- **Click** for the command-line interface framework
- **Sphinx / sphinx-book-theme / MyST-NB** for the documentation infrastructure
- **FHI-aims** — the electronic structure code that AimsPy drives, developed by the [aims team](https://aims-code.rg.mpg.de/)
- And many other essential libraries

### How to Acknowledge

When presenting work that uses AimsPy, please:

1. Cite our paper (see Citation section above)
2. Acknowledge the DeepH-pack team in presentations and publications
3. Consider contributing back improvements to benefit the community

## Getting Help and Contributing

### Support Channels

- **Documentation**: This documentation is your first resource
- **GitHub Issues**: For bug reports and feature requests
- **Examples Directory**: For practical implementation guides

### Ways to Contribute

We welcome contributions from everyone! Here's how you can help:

1. **Report bugs** - Help us identify issues
2. **Suggest features** - Share your ideas for improvements
3. **Improve documentation** - Fix typos, add examples, clarify explanations
4. **Submit code** - Fix bugs or implement new features
5. **Share examples** - Contribute scripts showcasing your use cases
6. **Help others** - Answer questions in the community

### Contribution Guidelines

Before contributing, please read our notes [For Developers](../for_developers/index) which includes:

- Code style and conventions
- Testing requirements
- Documentation standards
- Pull request process

---

## Final Notes

AimsPy is an ongoing project that continues to evolve with contributions from the materials science and computational physics communities. Your feedback, contributions, and use cases help shape the future development of this toolkit.

Whether you're using AimsPy for research, education, or industry applications, we hope it serves as a valuable tool in your computational materials science workflow.
