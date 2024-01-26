# pymatch

A simple, PyTorch-like automatic differentiation library for education.

Try running with

- [codon](https://github.com/exaloop/codon)
- [mojo](https://github.com/modularml/mojo)
- [numba](https://github.com/numba/numba)
- [cython](https://github.com/cython/cython)
- [pypy](https://www.pypy.org/)

Replace pure python `TensorData` with

- C extension (or pybind11, pyo3, nanobind)
- numpy

TODO:

- Add inplace operations in TensorData such as x.add_, x.radd_ ...
- Upgrade to 3.12, make venv, prof clark uses [miniforge](https://github.com/conda-forge/miniforge)
- Add note in readme about how to create my virtual environment

10/25

- Fix Types: use float|int instead of Union[float, int]
- Use lhs.reshape for a shallow copy instead of modifying lhs directly with lhs.reshape_; see line 630:635 in matmul tensordata.py
- Add white space for logically separable code blocks and comments
  - View three.js source-code coding style for reference

10/18
- Add white space for logically separable code blocks and comments
  - View three.js source-code coding style for reference
