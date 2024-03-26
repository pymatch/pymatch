from setuptools import Extension, setup

tensorbase_module = Extension(name="match.TensorBase", sources=["src/tensorbasemodule.c"])

setup(ext_modules=[tensorbase_module])
