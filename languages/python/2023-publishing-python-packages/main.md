# Python Packaging

Building a package requires a build system and build configs `pyproject.toml` or `setup.cfg`

Frontends: 
- pip
- build (pip install build or mamba install python-build). There are 2 commands to run, `python -m build` or `pyproject-build`

Backends: 
  - setuptools
  - hatchling

Build systems ignore files other than *.py. Therefore any file needed for the package should be included in `MANIFEST.in` file.
