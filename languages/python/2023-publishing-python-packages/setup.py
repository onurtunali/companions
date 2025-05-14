from Cython.Build import cythonize
from setuptools import setup

result = cythonize("src/flip/hello.py")

setup(ext_modules=result)
