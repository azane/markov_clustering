from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os


def do_setup():
    # Meant to be run from its own directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    setup(
        name="markov_clustering",
        ext_modules=cythonize("markov_clustering/c_mcl.pyx"),
        include_dirs=[numpy.get_include()]
    )

if __name__ == "__main__":
    do_setup()
