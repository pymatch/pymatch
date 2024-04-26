# Development Notes

## TODO

- [Dig into numba aot compilation](https://numba.readthedocs.io/en/stable/user/pycc.html)
- Dig into C-extension matmul speed ups (better caching?)
- Check C-extension for memory leaks (valgrind? python package?)
- look into [mojo resources](https://github.com/modularml/mojo/blob/main/examples/notebooks/Matmul.ipynb)
- [implement C-extension mapping](https://medium.com/@rcorbish/c-extensions-for-python3-731f4262f4b5)
- [Automatic stub generation](https://mypy.readthedocs.io/en/stable/stubgen.html)
- Try cython?

I'm not sure that we'll want to add this to the Python Package Index. The intended use is for reading through the code, not using as an installable library.

## Notes on C Extension

~~~bash
# Initial install (no --editable flag)
python -m pip install .

# Reinstall after changes (touch tensorbasemodule.c if editing tensorbase.c)
touch src/tensorbasemodule.c
python -m pip install --upgrade --force-reinstall .

# Testing
python -c "from match.tensorbase import randn, sigmoid; x = randn(2,2); sigmoid(x)"
~~~


## Alternative Python Implementations

We might need to run `match` using one of these alternative Python implementations to see if we can get a speedup.

- [codon](https://github.com/exaloop/codon)
- [mojo](https://github.com/modularml/mojo)
- [numba](https://github.com/numba/numba)
- [cython](https://github.com/cython/cython)
- [pypy](https://www.pypy.org/)
- [Mypyc](https://mypyc.readthedocs.io/en/latest/)

We could also replace the pure-python `TensorBase` with an extension or numpy (I don't like having numpy as an external dependency).

- C extension (or pybind11, pyo3, nanobind)
- numpy

**Strongly consider numba**: [The wrong way to speed up your code with Numba](https://pythonspeed.com/articles/slow-numba/)

Need to benchmark, ideally
- c extension
- codon
- numba
- pypy
- numpy

## Log/Notes

- Add in-place operations in `TensorBase` such as `x.add_`, `x.radd_`, etc.
- Upgrade to 3.12, create a virtual environment, prof clark uses [miniforge](https://github.com/conda-forge/miniforge)
- Add note in readme about how to create my virtual environment
- Update readme

10/25

- Fix Types: use float|int instead of Union[float, int]
- Use lhs.reshape for a shallow copy instead of modifying lhs directly with lhs.reshape_; see line 630:635 in matmul tensorbase.py
- Add white space for logically separable code blocks and comments
  - View three.js source-code coding style for reference

10/18

- Add white space for logically separable code blocks and comments
  - View three.js source-code coding style for reference
