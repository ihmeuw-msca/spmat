import numpy
from Cython.Build import cythonize
from setuptools import setup

if __name__ == "__main__":
    setup(
        ext_modules=cythonize("src/spmat/linalg.pyx"),
        include_dirs=[numpy.get_include()],
    )
