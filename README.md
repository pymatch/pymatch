# pymatch

A simple, PyTorch-like automatic differentiation library for education.

I'm not sure that we'll want to add this to the Python Package Index. The intended use is for reading through the code, not using as an installable library.

## Alternative Python Implementations

We might need to run `match` using one of these alternative Python implementations to see if we can get a speedup.

- [codon](https://github.com/exaloop/codon)
- [mojo](https://github.com/modularml/mojo)
- [numba](https://github.com/numba/numba)
- [cython](https://github.com/cython/cython)
- [pypy](https://www.pypy.org/)

We could also replace the pure-python `TensorData` with an extension or numpy (I don't like having numpy as an external dependency).

- C extension (or pybind11, pyo3, nanobind)
- numpy

## Log/Notes

TODO:

- Add in-place operations in `TensorData` such as `x.add_`, `x.radd_`, etc.
- Upgrade to 3.12, create a virtual environment, prof clark uses [miniforge](https://github.com/conda-forge/miniforge)
- Add note in readme about how to create my virtual environment

10/25

- Fix Types: use float|int instead of Union[float, int]
- Use lhs.reshape for a shallow copy instead of modifying lhs directly with lhs.reshape_; see line 630:635 in matmul tensordata.py
- Add white space for logically separable code blocks and comments
  - View three.js source-code coding style for reference

10/18

- Add white space for logically separable code blocks and comments
  - View three.js source-code coding style for reference
