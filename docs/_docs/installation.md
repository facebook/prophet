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

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/Configuring-C---Toolchain-for-Windows) provided by `rstan`. The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

<a href="#python"></a>

## Installation in Python

Prophet is on PyPI, so you can use `pip` to install it.

```bash
python -m pip install prophet
```

* From v0.6 onwards, Python 2 is no longer supported.
* As of v1.0, the package name on PyPI is "prophet"; prior to v1.0 it was "fbprophet".
* As of v1.1, the minimum supported Python version is 3.7.

After installation, you can [get started!](quick_start.html#python-api)

### Anaconda

Prophet can also be installed through conda-forge.

```bash
conda install -c conda-forge prophet
```
