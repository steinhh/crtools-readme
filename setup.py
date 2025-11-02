from setuptools import setup, Extension
from setuptools import find_packages
import numpy as np

extensions = [
    Extension(
        "crtools.fmedian._fmedian",
        sources=["crtools/fmedian/fmedian.c"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "crtools.fsigma._fsigma",
        sources=["crtools/fsigma/fsigma.c"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="crtools",
    version="0.0.0",
    description="CRTools: Cosmic Ray Removal Tools (C extensions)",
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=["numpy"],
)
