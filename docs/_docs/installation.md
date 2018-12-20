---
layout: docs
docid: 'installation'
title: 'Installation'
permalink: /docs/installation.html
subsections:
  - id: r
    title: Using R
  - id: python
    title: Using Python
---

Prophet has two implementations: [R](#installation-in-r) and [Python](#installation-in-python). Note the slight name difference for the Python package.

<a href="#r"></a>

## Installation in R

Due to [an upstream issue](https://github.com/r-lib/pkgbuild/issues/54) in pkgbuild, we recommend that you use `devtools` to install Prophet. This is currently the only way to get the latest release. We hope to resolve this in the near future.

```
# R
> devtools::install_github('facebook/prophet', subdir='R')
```

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) and you can continue to use `install.packages` if you are ok with using an earlier release:

```
# R
> install.packages('prophet')
```

After installation, you can [get started!](quick_start.html#r-api)

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Windows) provided by `rstan`. The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

<a href="#python"></a>

## Installation in Python

Prophet is on PyPI, so you can use pip to install it:

```
# bash
$ pip install fbprophet
```

The major dependency that Prophet has is `pystan`. PyStan has its own [installation instructions](http://pystan.readthedocs.io/en/latest/installation_beginner.html). Install pystan with pip before using pip to install fbprophet.

After installation, you can [get started!](quick_start.html#python-api)

### Windows

On Windows, PyStan requires a compiler so you'll need to [follow the instructions](http://pystan.readthedocs.io/en/latest/windows.html). The key step is installing a recent [C++ compiler](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Linux

Make sure compilers (gcc, g++, build-essential) and Python development tools (python-dev, python3-dev) are installed. In Red Hat systems, install the packages gcc64 and gcc64-c++. If you are using a VM, be aware that you will need at least 4GB of memory to install fbprophet, and at least 2GB of memory to use fbprophet.

### Anaconda

Use `conda install gcc` to set up gcc. The easiest way to install Prophet is through conda-forge: `conda install -c conda-forge fbprophet`.
