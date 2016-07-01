from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

import numpy as np

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    Extension(
        "bbox",
        ["faster_rcnn/bbox.pyx"],
        include_dirs=[numpy_include]
    ),
    Extension(
        "cpu_nms",
        ["faster_rcnn/cpu_nms.pyx"],
        include_dirs=[numpy_include]
    )
]

setup(
    name='fast_rcnn',
    ext_modules=cythonize(ext_modules),
)
