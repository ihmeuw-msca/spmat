# Special Matrices (spmat)

[![Build Status](https://github.com/ihmeuw-msca/spmat/workflows/python-build/badge.svg)](https://github.com/ihmeuw-msca/spmat/actions)
[![PyPI version](https://badge.fury.io/py/spmat.svg)](https://badge.fury.io/py/spmat)

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
