# Special Matrices (spmat)

[![PyPI](https://img.shields.io/pypi/v/spmat?color=purple)](https://pypi.org/project/spmat/)
![Python](https://img.shields.io/badge/python-3.10,_3.11,_3.12,_3.13-purple.svg)
[![Build](https://img.shields.io/github/actions/workflow/status/ihmeuw-msca/spmat/build.yml?branch=refactor/doc-cleanup&label=Build&color=purple)](https://github.com/ihmeuw-msca/spmat/actions)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/spmat?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=MAGENTA&left_text=Downloads)](https://pepy.tech/projects/spmat)

A collection of tools for special matrices with optimized implementations for scientific computing.

## Features

Currently includes:

- **`ILMat`**: Identity plus positive semi-definite (PSD) low-rank matrix
- **`DLMat`**: Diagonal plus positive semi-definite (PSD) low-rank matrix  
- **`BDLMat`**: Block diagonal plus low-rank matrix

## Installation

### From PyPI (Recommended)

```bash
pip install spmat
```

### From Source

Clone the repository and install:

```bash
git clone https://github.com/ihmeuw-msca/spmat.git
cd spmat
pip install -e .
```

## Requirements

- Python >= 3.10, < 3.14
- NumPy
- SciPy

## Development

To set up the development environment:

```bash
git clone https://github.com/ihmeuw-msca/spmat.git
cd spmat
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

This project is licensed under the BSD-2-Clause License - see the [LICENSE](LICENSE) file for details.
