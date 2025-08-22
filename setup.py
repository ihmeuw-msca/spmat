#!/usr/bin/env python3
from pathlib import Path

import numpy as np
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    
    # Read pyproject.toml
    with (base_dir / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    
    project = pyproject["project"]
    
    # Read long description from README
    with (base_dir / "README.md").open() as f:
        long_description = f.read()

    # Get requirements from pyproject.toml
    install_requirements = project["dependencies"]
    test_requirements = project["optional-dependencies"]["test"]
    doc_requirements = project["optional-dependencies"]["docs"]

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
        name=project["name"],
        version=project["version"],
        description=project["description"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        license=project["license"]["text"],
        url=project["urls"]["Homepage"],
        author=project["authors"][0]["name"],
        author_email=project["authors"][0]["email"],
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
