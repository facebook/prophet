# Prophet: Automatic Forecasting Procedure

Prophet is a procedure for forecasting time series data.  It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers.

Prophet is [open source software](https://code.facebook.com/projects/) released by Facebook's [Core Data Science team](https://research.fb.com/category/data-science/).  It is available for download on [CRAN](https://cran.r-project.org/package=prophet) and [PyPI](https://pypi.python.org/pypi/fbprophet/).

## Important links


- Homepage: https://facebook.github.io/prophet/
- HTML documentation: https://facebook.github.io/prophet/docs/quick_start.html
- Issue tracker: https://github.com/facebook/prophet/issues
- Source code repository: https://github.com/facebook/prophet
- Prophet R package: https://cran.r-project.org/package=prophet
- Prophet Python package: https://pypi.python.org/pypi/fbprophet/
- Release blogpost: https://research.fb.com/prophet-forecasting-at-scale/
- Prophet paper, "Forecasting at Scale": https://peerj.com/preprints/3190.pdf

## Installation in R

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) so you can use `install.packages`:

```
# R
> install.packages('prophet')
```

After installation, you can [get started!](https://facebook.github.io/prophet/docs/quick_start.html#r-api)

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Windows) provided by `rstan`.  The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

## Installation in Python

Prophet is on PyPI, so you can use pip to install it:

```
# bash
$ pip install fbprophet
```

The major dependency that Prophet has is `pystan`.   PyStan has its own [installation instructions](http://pystan.readthedocs.io/en/latest/installation_beginner.html).

After installation, you can [get started!](https://facebook.github.io/prophet/docs/quick_start.html#python-api)

### Windows

On Windows, PyStan requires a compiler so you'll need to [follow the instructions](http://pystan.readthedocs.io/en/latest/windows.html).  The key step is installing a recent [C++ compiler](http://landinghub.visualstudio.com/visual-cpp-build-tools).

### Linux

Make sure compilers (gcc, g++) and Python development tools (python-dev) are installed. If you are using a VM, be aware that you will need at least 4GB of memory to install fbprophet, and at least 2GB of memory to use fbprophet.

### Anaconda

Use `conda install gcc` to set up gcc. The easiest way to install Prophet is through conda-forge: `conda install -c conda-forge fbprophet`.

## Changelog

### Version 0.2.1 (2017.11.08)

- Bugfixes

### Version 0.2 (2017.09.02)

- Forecasting with sub-daily data
- Daily seasonality, and custom seasonalities
- Extra regressors
- Access to posterior predictive samples
- Cross-validation function
- Saturating minimums
- Bugfixes

### Version 0.1.1 (2017.04.17)

- Bugfixes
- New options for detecting yearly and weekly seasonality (now the default)

### Version 0.1 (2017.02.23)

- Initial release
