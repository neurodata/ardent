import sys
from setuptools import setup, find_packages
from pathlib import Path

def check_python_version():
    """Raises SystemExit when the Python version is too low."""
    if sys.version_info < version_info['__min_python_version__']:
        raise SystemExit((f"Python >= {version_info['__min_python_version__'][0]}.{version_info['__min_python_version__'][1]} is required."
            f"\nYou are running Python {sys.version_info[0]}.{sys.version_info[1]}."))
check_python_version()

# Variables defined in __version__.py.
version_info = {}
with open(Path(__file__).parent / 'ardent/__version__.py', 'r') as versionFile:
    exec(versionFile.read(), version_info)

# Descriptive text contained in README.md.
with open('README.md', 'r') as readmeFile:
    README = readmeFile.read()

setup(
    name=version_info['__title__'],
    description=version_info['__description__'],
    long_description=README,
    long_description_content_type='text/markdown',
    version=version_info['__version__'],
    url=version_info['__url__'],
    author=version_info['__author__'],
    author_email=version_info['__author_email__'],
    python_requires=f">={version_info['__min_python_version__'][0]}.{version_info['__min_python_version__'][1]}",
    install_requires=version_info['__required_packages__'],
    license=version_info['__license__'],
    classifiers=version_info['__classifiers__'],
    packages=find_packages(),
)