import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np

MODULE_NAME = 'pystreamline'

module_streamline = Extension(
    MODULE_NAME + '.streamline_wrapper',
    include_dirs=[os.path.join(MODULE_NAME, 'src'), np.get_include()],
    sources=[
        os.path.join(MODULE_NAME, rel_path) for rel_path in [
            'streamline_wrapper.pyx',
            'src/streamline.cpp',
            'src/kdtree.cpp',
        ]
    ],
    language='c++',
    extra_compile_args=['--std=c++11', '-g'],
    extra_link_args=["-g"],
)

setup(
    name=MODULE_NAME,
    version='0.1',
    description='Calculates streamlines for vector fields.',
    zip_safe=False,
    packages=find_packages(),
    setup_requires=['numpy>=1.14.5', 'Cython>=0.28.3',],
    install_requires=['numpy>=1.14.5', 'Cython>=0.28.3',],
    tests_require=['pytest>=3.6.2',],
    ext_modules=cythonize([module_streamline]),
    cmdclass = {'build_ext': build_ext},
    author='Chris Knight',
    author_email='chrisk314@gmail.com',
)
