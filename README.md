# Prophet: Automatic Forecasting Procedure

![Build](https://github.com/facebook/prophet/workflows/Build/badge.svg)
[![Pypi_Version](https://img.shields.io/pypi/v/fbprophet.svg)](https://pypi.python.org/pypi/fbprophet)
[![Conda_Version](https://anaconda.org/conda-forge/fbprophet/badges/version.svg)](https://anaconda.org/conda-forge/fbprophet/)

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Prophet is [open source software](https://code.facebook.com/projects/) released by Facebook's [Core Data Science team](https://research.fb.com/category/data-science/). It is available for download on [CRAN](https://cran.r-project.org/package=prophet) and [PyPI](https://pypi.python.org/pypi/fbprophet/).

## Important links

- Homepage: https://facebook.github.io/prophet/
- HTML documentation: https://facebook.github.io/prophet/docs/quick_start.html
- Issue tracker: https://github.com/facebook/prophet/issues
- Source code repository: https://github.com/facebook/prophet
- Contributing: https://facebook.github.io/prophet/docs/contributing.html
- Prophet R package: https://cran.r-project.org/package=prophet
- Prophet Python package: https://pypi.python.org/pypi/fbprophet/
- Release blogpost: https://research.fb.com/prophet-forecasting-at-scale/
- Prophet paper: Sean J. Taylor, Benjamin Letham (2018) Forecasting at scale. The American Statistician 72(1):37-45 (https://peerj.com/preprints/3190.pdf).

## Installation in R

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) so you can use `install.packages`.

```R
install.packages('prophet')
```

After installation, you can [get started!](https://facebook.github.io/prophet/docs/quick_start.html#r-api)

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started) provided by `rstan`. The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

## Installation in Python

Prophet is on PyPI, so you can use `pip` to install it. From v0.6 onwards, Python 2 is no longer supported.

```bash
# Install pystan with pip before using pip to install fbprophet
pip install pystan

pip install fbprophet
```

The default dependency that Prophet has is `pystan`. PyStan has its own [installation instructions](http://pystan.readthedocs.io/en/latest/installation_beginner.html). Install pystan with pip before using pip to install fbprophet.

You can also choose a (more experimental) alternative stan backend called `cmdstanpy`. It requires the [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) command line interface and you will have to specify the environment variable `STAN_BACKEND` pointing to it, for example:

```
# bash
$ CMDSTAN=/tmp/cmdstan-2.22.1 STAN_BACKEND=CMDSTANPY pip install fbprophet
```

Note that the `CMDSTAN` variable is directly related to `cmdstanpy` module and can be omitted if your CmdStan binaries are in your `$PATH`.

It is also possible to install Prophet with two backends:

```
# bash
$ CMDSTAN=/tmp/cmdstan-2.22.1 STAN_BACKEND=PYSTAN,CMDSTANPY pip install fbprophet
```

After installation, you can [get started!](https://facebook.github.io/prophet/docs/quick_start.html#python-api)

If you upgrade the version of PyStan installed on your system, you may need to reinstall fbprophet ([see here](https://github.com/facebook/prophet/issues/324)).

### Anaconda

Use `conda install gcc` to set up gcc. The easiest way to install Prophet is through conda-forge: `conda install -c conda-forge fbprophet`.

### Windows

On Windows, PyStan requires a compiler so you'll need to [follow the instructions](http://pystan.readthedocs.io/en/latest/windows.html). The easiest way to install Prophet in Windows is in Anaconda.

### Linux

Make sure compilers (gcc, g++, build-essential) and Python development tools (python-dev, python3-dev) are installed. In Red Hat systems, install the packages gcc64 and gcc64-c++. If you are using a VM, be aware that you will need at least 4GB of memory to install fbprophet, and at least 2GB of memory to use fbprophet.

## Changelog

### Version 0.6 (2020.03.03)

- Fix bugs related to upstream changes in `holidays` and `pandas` packages.
- Compile model during first use, not during install (to comply with CRAN policy)
- `cmdstanpy` backend now available in Python
- Python 2 no longer supported

### Version 0.5 (2019.05.14)

- Conditional seasonalities
- Improved cross validation estimates
- Plotly plot in Python
- Bugfixes

### Version 0.4 (2018.12.18)

- Added holidays functionality
- Bugfixes

### Version 0.3 (2018.06.01)

- Multiplicative seasonality
- Cross validation error metrics and visualizations
- Parameter to set range of potential changepoints
- Unified Stan model for both trend types
- Improved future trend uncertainty for sub-daily data
- Bugfixes

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

## License

Prophet is licensed under the [MIT license](LICENSE).
