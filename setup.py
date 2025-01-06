from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys
import numpy as np

# Get Eigen include directory
eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", "/usr/include/eigen3")

ext_modules = [
    Pybind11Extension(
        "genetic_algorithm",  # Module name
        ["src/GA.cpp", "src/ga_bindings.cpp"],  # Source files
        include_dirs=[
            "include",  # Local include directory
            eigen_include_dir,  # Eigen include directory
            np.get_include(),  # NumPy include directory
        ],
        libraries=["yaml-cpp"],  # Libraries to link against
    ),
]

setup(
    name="genetic_algorithm",
    version="1.0",
    description="Genetic Algorithm with C++ and Python bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
