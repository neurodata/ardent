import sys
from setuptools import setup, find_packages
from pathlib import Path

# whose my daddy
parent_ = Path(__file__).parent

# Variables defined in __version__.py.
version_info = {}
version_file = parent_ / 'ardent/__version__.py'
with open(version_file) as VersionFile:
    exec(VersionFile.read(), version_info)

# Descriptive text contained in README.md.
with open('README.md', 'r') as ReadmeFile:
    README = ReadmeFile.read()

# Requirements defined in requirements.txt.
requirements_file = parent_ / 'requirements.txt'
with open(requirements_filie, 'r') as RequirementsFile:
    install_requires = RequirementsFile.readlines()

# I think you can do this implicitly in `setup.py`
def check_python_version():
    """Raises SystemExit when the Python version is too low."""
    if sys.version_info < version_info['__min_python_version__']:
        raise SystemExit((f"Python >= {version_info['__min_python_version__'][0]}.{version_info['__min_python_version__'][1]} is required."
            f"\nYou are running Python {sys.version_info[0]}.{sys.version_info[1]}."))
check_python_version()

setup(
    name=version_info['__name__'],
    description=version_info['__description__'],
    long_description=README,
    long_description_content_type='text/markdown',
    version=version_info['__version__'],
    url=version_info['__url__'],
    author=version_info['__author__'],
    author_email=version_info['__author_email__'],
    python_requires=version_info['__python_requires__'],
    license=version_info['__license__'],
    classifiers=version_info['__classifiers__'],
    install_requires=install_requires,
    packages=find_packages(),
)
