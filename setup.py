from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='cluster_colors',
    ext_modules = cythonize("cluster_colors/stack_colors.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
