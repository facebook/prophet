# Prophet: Automatic Forecasting Procedure

![Build](https://github.com/facebook/prophet/workflows/Build/badge.svg)

[![PyPI Version](https://img.shields.io/pypi/v/prophet.svg)](https://pypi.python.org/pypi/prophet)
[![PyPI Downloads Monthly](https://pepy.tech/badge/prophet/month)](https://pepy.tech/project/prophet)
[![PyPI Downloads All](https://pepy.tech/badge/prophet)](https://pepy.tech/project/prophet)

[![CRAN Version](https://www.r-pkg.org/badges/version/prophet)](https://CRAN.R-project.org/package=prophet)
[![CRAN Downloads Monthly](https://cranlogs.r-pkg.org/badges/prophet?color=brightgreen)](https://cran.r-project.org/package=prophet)
[![CRAN Downloads All](https://cranlogs.r-pkg.org/badges/grand-total/prophet?color=brightgreen)](https://cranlogs.r-pkg.org/badges/grand-total/prophet)

[![Conda_Version](https://anaconda.org/conda-forge/prophet/badges/version.svg)](https://anaconda.org/conda-forge/prophet/)

Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

Prophet is [open source software](https://code.facebook.com/projects/) released by Facebook's [Core Data Science team](https://research.fb.com/category/data-science/). It is available for download on [CRAN](https://cran.r-project.org/package=prophet) and [PyPI](https://pypi.python.org/pypi/prophet/).

## Important links

- Homepage: https://facebook.github.io/prophet/
- HTML documentation: https://facebook.github.io/prophet/docs/quick_start.html
- Issue tracker: https://github.com/facebook/prophet/issues
- Source code repository: https://github.com/facebook/prophet
- Contributing: https://facebook.github.io/prophet/docs/contributing.html
- Prophet R package: https://cran.r-project.org/package=prophet
- Prophet Python package: https://pypi.python.org/pypi/prophet/
- Release blogpost: https://research.fb.com/prophet-forecasting-at-scale/
- Prophet paper: Sean J. Taylor, Benjamin Letham (2018) Forecasting at scale. The American Statistician 72(1):37-45 (https://peerj.com/preprints/3190.pdf).

## Installation in R

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) so you can use `install.packages`.

```r
install.packages('prophet')
```

After installation, you can [get started!](https://facebook.github.io/prophet/docs/quick_start.html#r-api)

#### Experimental backend - cmdstanr

You can also choose an experimental alternative stan backend called `cmdstanr`. Once you've installed `prophet`,
follow these instructions to use `cmdstanr` instead of `rstan` as the backend:

```r
# R
# We recommend running this in a fresh R session or restarting your current session
install.packages(c("cmdstanr", "posterior"), repos = c("https://mc-stan.org/r-packages/", getOption("repos")))

# If you haven't installed cmdstan before, run:
cmdstanr::install_cmdstan()
# Otherwise, you can point cmdstanr to your cmdstan path:
cmdstanr::set_cmdstan_path(path = <your existing cmdstan>)

# Set the R_STAN_BACKEND environment variable
Sys.setenv(R_STAN_BACKEND = "CMDSTANR")
```

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started) provided by `rstan`. The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

If you have custom Stan compiler settings, install from source rather than the CRAN binary.

## Installation in Python - PyPI release

Prophet is on PyPI, so you can use `pip` to install it.

```bash
python -m pip install prophet
```

* From v0.6 onwards, Python 2 is no longer supported.
* As of v1.0, the package name on PyPI is "prophet"; prior to v1.0 it was "fbprophet".
* As of v1.1, the minimum supported Python version is 3.7.

After installation, you can [get started!](https://facebook.github.io/prophet/docs/quick_start.html#python-api)

### Anaconda

Prophet can also be installed through conda-forge: `conda install -c conda-forge prophet`.

## Installation in Python - Development version

To get the latest code changes as they are merged, you can clone this repo and build from source manually. This is **not** guaranteed to be stable.

```bash
git clone https://github.com/facebook/prophet.git
cd prophet/python
python -m pip install -r requirements.txt
python setup.py develop
```

By default, Prophet will use a fixed version of `cmdstan` (downloading and installing it if necessary) to compile the model executables. If this is undesired and you would like to use your own existing `cmdstan` installation, you can set the environment variable `PROPHET_REPACKAGE_CMDSTAN` to `False`:

```bash
export PROPHET_REPACKAGE_CMDSTAN=False; python setup.py develop
```

### Linux

Make sure compilers (gcc, g++, build-essential) and Python development tools (python-dev, python3-dev) are installed. In Red Hat systems, install the packages gcc64 and gcc64-c++. If you are using a VM, be aware that you will need at least 4GB of memory to install prophet, and at least 2GB of memory to use prophet.

### Windows

Using `cmdstanpy` with Windows requires a Unix-compatible C compiler such as mingw-gcc. If cmdstanpy is installed first, one can be installed via the `cmdstanpy.install_cxx_toolchain` command.

## Changelog

### Version 1.1.1 (2022.09.08)

- (Python) Improved runtime (3-7x) of uncertainty predictions via vectorization.
- Bugfixes relating to Python package versions and R holiday objects.

### Version 1.1 (2022.06.25)

- Replaced `pystan2` dependency with `cmdstan` + `cmdstanpy`.
- Pre-packaged model binaries for Python package, uploaded binary distributions to PyPI.
- Improvements in the `stan` model code, cross-validation metric calculations, holidays.

### Version 1.0 (2021.03.28)

- Python package name changed from fbprophet to prophet
- Fixed R Windows build issues to get latest version back on CRAN
- Improvements in serialization, holidays, and R timezone handling
- Plotting improvements

### Version 0.7 (2020.09.05)

- Built-in json serialization
- Added "flat" growth option
- Bugfixes related to `holidays` and `pandas`
- Plotting improvements
- Improvements in cross validation, such as parallelization and directly specifying cutoffs

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
