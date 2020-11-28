from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "im2col_cython", ["CNN/cs231n_fc_nets/im2col_cython.pyx"], include_dirs=[numpy.get_include()]
    ),
]

setup(ext_modules=cythonize(extensions),)