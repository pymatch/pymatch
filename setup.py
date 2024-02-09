from setuptools import Extension, setup

tensordata_module = Extension("match.tensordata", sources=["src/tensordata.c"])

setup(ext_modules=[tensordata_module])
