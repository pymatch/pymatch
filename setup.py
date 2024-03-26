from setuptools import Extension, setup

tensordata_module = Extension("match.tensorbase", sources=["src/tensorbase.c"])

setup(ext_modules=[tensordata_module])
