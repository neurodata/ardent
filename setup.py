import sys
from setuptools import setup, find_packages
from pathlib import Path

# Variables defined in __version__.py.
version_info = {}
with open(Path(__file__) / '__version__.py', 'r') as versionFile:
    exec(versionFile.read(), version_info)

# Text contained in README.md.
with open('README.md', 'r') as readmeFile:
    README = readmeFile.read()

def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        raise SystemExit((f"Python {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]} is required."
            f"\nYou are running Python {sys.version_info[0]}.{sys.version_info[1]}."))
check_python_version()

setup(
    name=version_info['__title__'],
    description=version_info['__description__'],
    long_description=README,
    long_description_content_type='text/markdown',
    version=version_info['__version__'],
    url=version_info['__url__'],
    author=version_info['__author__'],
    author_email=version_info['__author_email__'],
    python_requires=version_info['__python_requires__'],
    install_requires=version_info['__required_packages__'],
    license=version_info['__license__'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
)