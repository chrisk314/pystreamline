# Py Streamline
Python wrapper around a C++ streamline integrator.

### Installation
This module requires Cython. Currently, Cython needs to be installed before
running the installation procedure for this module due to the use of Cython
components in the  `setup.py` script. This
[issue](https://github.com/pypa/setuptools/issues/1317) is mentioned on the
setuptools repo.

To install run the below commands

```
git clone git@github.com:chrisk314/pystreamline.git
cd pystreamline
pip install -r requirements.in  # Ensures Cython and numpy are installed
pip install .
```

### Calculating streamlines
TODO
