import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize("src/spmat/linalg.pyx"),
    include_dirs=[numpy.get_include()],
)
