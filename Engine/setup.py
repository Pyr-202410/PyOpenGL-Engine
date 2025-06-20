from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name='BaseEngine',
    ext_modules=cythonize("BaseEngine.py"),
    include_dirs=[numpy.get_include()],
    requires=['glfw', 'pyopengl', 'glm', 'numpy', 'Pillow']
)