#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src" / "spmat"

    # Read long description from README
    with (base_dir / "README.md").open() as f:
        long_description = f.read()

    # Define requirements
    install_requirements = ["numpy", "scipy"]
    test_requirements = ["pytest", "pytest-mock"]
    doc_requirements = []

    def get_extensions():
        """Get Cython extensions."""
        return [
            Extension(
                "spmat.linalg",
                sources=["src/spmat/linalg.pyx"],
                include_dirs=[np.get_include()],
                libraries=["m"],
            )
        ]

    setup(
        name="spmat",
        version="0.0.12",
        description="A collection of tools for special matrices",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="BSD 2-Clause License",
        url="https://github.com/zhengp0/spmat",
        author="Peng Zheng",
        author_email="zhengp@uw.edu",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        include_package_data=True,
        install_requires=install_requirements,
        tests_require=test_requirements,
        extras_require={
            "docs": doc_requirements,
            "test": test_requirements,
            "dev": doc_requirements + test_requirements,
        },
        ext_modules=cythonize(
            get_extensions(),
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
            },
        ),
        zip_safe=False,
    )
