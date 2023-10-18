# pymatch
 A pure-Python, PyTorch-like automatic differentiation library for education. Topics

 TODO:
- implement matrix multiplication: https://pytorch.org/docs/stable/generated/torch.matmul.html [DONE]
- add inplace operations in tensordata x.add_, x.radd_ ...
- upgrade to 3.12, make venv, prof clark uses mamba forge: https://github.com/conda-forge/miniforge#Mambaforge
- fix the reshape_ and reshape functions to handle singletons [DONE]
reshape should turn singleton (shape () into shape(1)), and should turn a shape (1) into shape () l.reshape(()), back into a singleton
- also, reshape should handle this functionality [DONE]
>>> l
tensor([1.])
>>> j
tensor(1.)
>>> l[0] = 5
>>> l
tensor([5.])
>>> j
tensor(5.)
>>>


- Add note in readme about how to create my virtual environment
