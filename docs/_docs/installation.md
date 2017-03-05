---
layout: docs
docid: "installation"
title: "Installation"
permalink: /docs/installation.html
---

Prophet has two implementations: [R](#installation-in-r) and [Python](#installation-in-python).  Note the slight name difference for the Python package.

## Installation in R

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) so you can use `install.packages`:

```
# R
> install.packages('prophet')
```

After installation, you can [get started!](quick_start.html#r-api)

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Windows) provided by `rstan`.  The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

## Installation in Python

Prophet is on PyPI, so you can use pip to install it:

```
# bash
$ pip install fbprophet
```

The major dependency that Prophet has is `pystan`.   PyStan has its own [installation instructions](http://pystan.readthedocs.io/en/latest/installation_beginner.html).

After installation, you can [get started!](quick_start.html#python-api)

### Windows

On Windows, PyStan requires a compiler so you'll need to [follow the instructions](http://pystan.readthedocs.io/en/latest/windows.html).  The key step is installing a recent [C++ compiler](http://landinghub.visualstudio.com/visual-cpp-build-tools).

### Linux VM

If you want to use Prophet in a VM, you will need at least 2GB of memory to run PyStan.
