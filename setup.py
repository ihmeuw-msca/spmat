import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[Extension("spmat.linalg", ["src/spmat/linalg.pyx"])],
    include_dirs=[numpy.get_include()],
)
