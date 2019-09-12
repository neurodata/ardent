# ARDENT
**A**ffine and **R**egularized **DE**formative **N**umeric **T**ransform (ARDENT) is a Python package for performing automated image registration using LDDMM.

ARDENT stands out for its ability to predict and correct for artifacts and image nonuniformity, perform registrations across image modalities, ease of use, and other features in development.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](#license)

# Overview
Experimental neuroscience produces a stunning amount of imaging data from light or electron microscopy, MRI, and other 3D modalities. To be of real use these datasets must be interpreted with respect to each other and to refined standards: well-characterized image datasets called atlases. To build these interpretations, dense spatial alignments must be computed. This process is known as image registration, in which one image is optimally deformed, or flowed, until it aligns with another. Accurate registration is challenged by the large scale of imaging data and the heterogeneity across species scales and modalities. Current tools can perform well on very standard images but perform poorly on data with various imperfections. This restricts our ability to analyze data from novel experiments performed in a majority of labs.

ARDENT is an accessible pure-python image registration package in development with these neuroimaging challenges in mind.

# Documentation
The official documentation with usage is at https://ardent.neurodata.io/

Please visit the [tutorial section](https://ardent.neurodata.io/tutorial.html) in the official website for more in-depth usage.

# System requirements

## Hardware requirements
`ARDENT` package requires only a standard computer with enough RAM to support the in-memory operations.

### Python Requirements
This package is written for Python3. Currently, it is supported for Python 3.6.

### Python Dependencies
`ARDENT` mainly depends on the Python scientific stack with the notable addition of PyTorch. However, this is pending deprecation.
```
numpy
matplotlib
scipy
scikit-learn
simpleitk
nibabel
nilearn
pytorch
```

# Installation Guide
## Install from pip
```
pip install ardent
```

## Install from Github
```
git clone https://github.com/neurodata/ardent
cd ardent
python3 setup.py install
```

# License
This project is covered under the [Apache 2.0 License](https://github.com/neurodata/ardent/blob/master/LICENSE).
