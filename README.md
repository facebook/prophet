# Prophet: Automatic Forecasting Procedure

Prophet is a procedure for forecasting time series data.  It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers.

Prophet is [open source software](https://code.facebook.com/projects/) released by Facebook's [Core Data Science team](https://research.fb.com/category/data-science/).  It is available for download on [CRAN](https://cran.r-project.org/package=prophet) and [PyPI](https://pypi.python.org/pypi/fbprophet/).

## Important links


- Homepage: https://facebookincubator.github.io/prophet/
- HTML documentation: https://facebookincubator.github.io/prophet/docs/quick_start.html
- Issue tracker: https://github.com/facebookincubator/prophet/issues
- Source code repository: https://github.com/facebookincubator/prophet
- Prophet R package: https://cran.r-project.org/package=prophet
- Prophet Python package: https://pypi.python.org/pypi/fbprophet/

## Installation in R

Prophet is a [CRAN package](https://cran.r-project.org/package=prophet) so you can use `install.packages`:

```
# R
> install.packages('prophet')
```

After installation, you can [get started!](https://facebookincubator.github.io/prophet/docs/quick_start.html#r-api)

### Windows

On Windows, R requires a compiler so you'll need to [follow the instructions](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Windows) provided by `rstan`.  The key step is installing [Rtools](http://cran.r-project.org/bin/windows/Rtools/) before attempting to install the package.

## Installation in Python

Prophet is on PyPI, so you can use pip to install it:

```
# bash
$ pip install fbprophet
```

The major dependency that Prophet has is `pystan`.   PyStan has its own [installation instructions](http://pystan.readthedocs.io/en/latest/installation_beginner.html).

After installation, you can [get started!](https://facebookincubator.github.io/prophet/docs/quick_start.html#python-api)

### Windows

On Windows, PyStan requires a compiler so you'll need to [follow the instructions](http://pystan.readthedocs.io/en/latest/windows.html).  The key step is installing a recent [C++ compiler](http://landinghub.visualstudio.com/visual-cpp-build-tools).

