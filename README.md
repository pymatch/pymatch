# pymatch
 A pure-Python, PyTorch-like automatic differentiation library for education. Topics

How to run single test: python3 -m unittest tests.test_tensor.TestTensor.test_matmul_nd_1d

 TODO:
- Add inplace operations in TensorData such as x.add_, x.radd_ ...
- Upgrade to 3.12, make venv, prof clark uses mamba forge: https://github.com/conda-forge/miniforge#Mambaforge
- Add note in readme about how to create my virtual environment

10/25
- Fix Types: use float|int instead of Union[float, int]
- Use lhs.reshape for a shallow copy instead of modifying lhs directly with lhs.reshape_; see line 630:635 in matmul tensordata.py
- Add white space for logically separable code blocks and comments
       - View three.js source-code coding style for reference

10/18
- Add white space for logically separable code blocks and comments
       - View three.js source-code coding style for reference


3/19
- 
