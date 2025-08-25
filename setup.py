import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src" / "spmat"

    sys.path.insert(0, src_dir.as_posix())
    import __about__ as about



    setup(
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        ext_modules=cythonize("src/spmat/linalg.pyx"),
        include_dirs=[numpy.get_include()],
        zip_safe=False,
    )
