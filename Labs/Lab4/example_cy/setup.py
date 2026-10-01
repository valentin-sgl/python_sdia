from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("helloworld", ["helloworld.pyx"]),
    Extension("primes", ["primes.pyx"])
]

setup(
    ext_modules=cythonize(extensions, annotate=True, language_level="3")
)