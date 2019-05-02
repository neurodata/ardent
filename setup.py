import sys
from setuptools import setup, find_packages
from pathlib import Path

PACKAGE_NAME = "ardent"
DESCRIPTION = "A tool for image registration."
with open("README.md", "r") as readmeFile:
    LONG_DESCRIPTION = readmeFile.read()
AUTHOR = "Devin Crowley"
AUTHOR_EMAIL = "devin.g.crowley@gmail.com"
URL = "https://github.com/neurodata/ardent"
LICENSE = "Eclipse Public License 2.0"
MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.6
REQUIRED_PACKAGES = [
    "numpy",
    "pathlib",
    "pytorch",
]

# Find ardent version.
PROJECT_PATH = str(Path(__file__).parent)
for line in open(Path(PROJECT_PATH) / "ardent" / "__init__.py"):
    if line.startswith("__version__ = "):
    VERSION = line.strip().split()[2][1:-1] ##################################################### INVESTIGATE


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        raise SystemExit(f"Python {MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1] is required. \
            You are running Python {sys.version_info[0]}.{sys.version_info[1]}.")


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license=LICENSE,
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