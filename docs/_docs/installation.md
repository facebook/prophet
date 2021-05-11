---
layout: docs
docid: "installation"
title: "Installation"
permalink: /docs/installation.html
subsections:
  - id: r
    title: Using R
  - id: python
    title: Using Python
---

Prophet has two implementations: [R](#installation-in-r) and [Python](#installation-in-python).

<a href="#r"></a>

## Installation in R

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) so you can use `install.packages`.

```r
# R
install.packages('prophet')
```

After installation, you can [get started!](quick_start.html#r-api)

#### Experimental backend - cmdstanr

You can also choose an experimental alternative stan backend called `cmdstanr`. Once you've installed `prophet`,
follow these instructions to use `cmdstanr` instead of `rstan` as the backend:

```r
# R
# We recommend running this is a fresh R session or restarting your current session
install.packages(c("cmdstanr", "posterior"), repos = c("https://mc-stan.org/r-packages/", getOption("repos")))

# If you haven't installed cmdstan before, run:
cmdstanr::install_cmdstan()
# Otherwise, you can point cmdstanr to your cmdstan path:
cmdstanr::set_cmdstan_path(path = <your existing cmdstan>)

# Set the R_STAN_BACKEND environment variable
Sys.setenv(R_STAN_BACKEND = "CMDSTANR")
```

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Windows) provided by `rstan`. The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

<a href="#python"></a>

## Installation in Python

Prophet is on PyPI, so you can use pip to install it:

```bash
# bash
# Install pystan with pip before using pip to install prophet
# pystan>=3.0 is currently not supported
$ pip install pystan==2.19.1.1
$
$ pip install prophet
```

The major dependency that Prophet has is `pystan`. PyStan has its own [installation instructions](https://pystan.readthedocs.io/en/latest/installation.html). Install pystan with pip before using pip to install prophet.

If you upgrade the version of PyStan installed on your system, you may need to reinstall prophet ([see here](https://github.com/facebook/prophet/issues/324)).

After installation, you can [get started!](quick_start.html#python-api)

#### Experimental backend - cmdstanpy

You can also choose a (more experimental) alternative stan backend called cmdstanpy. It requires the CmdStan command line interface and you will have to specify the environment variable STAN_BACKEND pointing to it, for example:

```bash
# bash
$ CMDSTAN=/tmp/cmdstan-2.22.1 STAN_BACKEND=CMDSTANPY pip install prophet
```

Note that the CMDSTAN variable is directly related to cmdstanpy module and can be omitted if your CmdStan binaries are in your $PATH.

It is also possible to install Prophet with two backends:

```bash
# bash
$ CMDSTAN=/tmp/cmdstan-2.22.1 STAN_BACKEND=PYSTAN,CMDSTANPY pip install prophet
```

### Windows

On Windows, PyStan requires a compiler so you'll need to [follow the instructions](http://pystan.readthedocs.io/en/latest/windows.html).  The easiest way to install Prophet in Windows is in Anaconda.

### Linux

Make sure compilers (gcc, g++, build-essential) and Python development tools (python-dev, python3-dev) are installed. In Red Hat systems, install the packages gcc64 and gcc64-c++. If you are using a VM, be aware that you will need at least 4GB of memory to install prophet, and at least 2GB of memory to use prophet.

### Anaconda

Use `conda install gcc` to set up gcc. The easiest way to install Prophet is through conda-forge: `conda install -c conda-forge prophet`.
