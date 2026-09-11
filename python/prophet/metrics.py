# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Final, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import TypeAlias, TypeVar

    _PerformanceMetric: TypeAlias = Callable[[pd.DataFrame, int], pd.DataFrame]
    _PerformanceMetricT = TypeVar("_PerformanceMetricT", bound=_PerformanceMetric)


logger: logging.Logger = logging.getLogger('prophet')

PERFORMANCE_METRICS: Final[dict[str, Callable[..., Any]]] = {}


def register_performance_metric(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register custom performance metric

    Parameters that your metric should contain
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Registered metric should return following
    -------
    Dataframe with columns horizon and metric.
    """
    PERFORMANCE_METRICS[func.__name__] = func
    return func


def performance_metrics(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    rolling_window: float = 0.1,
    monthly: bool = False,
) -> pd.DataFrame | None:
    """Compute performance metrics from cross-validation results.

    Computes a suite of performance metrics on the output of cross-validation.
    By default, the following metrics are included:
    'mse': mean squared error
    'rmse': root mean squared error
    'mae': mean absolute error
    'mape': mean absolute percent error
    'mdape': median absolute percent error
    'smape': symmetric mean absolute percentage error
    'coverage': coverage of the upper and lower intervals

    A subset of these can be specified by passing a list of names as the
    `metrics` argument.

    Metrics are calculated over a rolling window of cross validation
    predictions, after sorting by horizon. Averaging is first done within each
    value of horizon, and then across horizons as needed to reach the window
    size. The size of that window (number of simulated forecast points) is
    determined by the rolling_window argument, which specifies a proportion of
    simulated forecast points to include in each window. rolling_window=0 will
    compute it separately for each horizon. The default of rolling_window=0.1
    will use 10% of the rows in df in each window. rolling_window=1 will
    compute the metric across all simulated forecast points. The results are
    set to the right edge of the window.

    If rolling_window < 0, then metrics are computed at each datapoint with no
    averaging (i.e., 'mse' will actually be squared error with no mean).

    The output is a dataframe containing column 'horizon' along with columns
    for each of the metrics computed.

    Parameters
    ----------
    df: The dataframe returned by cross_validation.
    metrics: A list of performance metrics to compute. If not provided, will
        use ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'].
    rolling_window: Proportion of data to use in each rolling window for
        computing the metrics. Should be in [0, 1] to average.
    monthly: monthly=True will compute horizons as numbers of calendar months
        from the cutoff date, starting from 0 for the cutoff month.

    Returns
    -------
    Dataframe with a column for each metric, and column 'horizon'
    """
    valid_metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']
    if metrics is None:
        metrics = valid_metrics
    if ('yhat_lower' not in df or 'yhat_upper' not in df) and ('coverage' in metrics):
        metrics.remove('coverage')
    if len(set(metrics)) != len(metrics):
        raise ValueError('Input metrics must be a list of unique values')
    if not set(metrics).issubset(set(PERFORMANCE_METRICS)):
        raise ValueError(
            'Valid values for metrics are: {}'.format(valid_metrics)
        )
    df_m = df.copy()
    if monthly:
        df_m['horizon'] = df_m['ds'].dt.to_period('M').astype(int) - df_m['cutoff'].dt.to_period('M').astype(int)
    else:
        df_m['horizon'] = df_m['ds'] - df_m['cutoff']
    df_m.sort_values('horizon', inplace=True)
    if 'mape' in metrics and df_m['y'].abs().min() < 1e-8:
        logger.info('Skipping MAPE because y close to 0')
        metrics.remove('mape')
    if len(metrics) == 0:
        return None
    w = int(rolling_window * df_m.shape[0])
    if w >= 0:
        w = max(w, 1)
        w = min(w, df_m.shape[0])
    # Compute all metrics
    dfs = {}
    for metric in metrics:
        dfs[metric] = PERFORMANCE_METRICS[metric](df_m, w)
    res = dfs[metrics[0]]
    for i in range(1, len(metrics)):
        res_m = dfs[metrics[i]]
        assert np.array_equal(res['horizon'].values, res_m['horizon'].values)
        res[metrics[i]] = res_m[metrics[i]]
    return res


def rolling_mean_by_h(x: np.ndarray, h: np.ndarray, w: int, name: str) -> pd.DataFrame:
    """Compute a rolling mean of x, after first aggregating by h.

    Right-aligned. Computes a single mean for each unique value of h. Each
    mean is over at least w samples.

    Parameters
    ----------
    x: Array.
    h: Array of horizon for each value in x.
    w: Integer window size (number of elements).
    name: Name for metric in result dataframe

    Returns
    -------
    Dataframe with columns horizon and name, the rolling mean of x.
    """
    # Aggregate over h
    df = pd.DataFrame({'x': x, 'h': h})
    df2 = (
        df.groupby('h').agg(['sum', 'count']).reset_index().sort_values('h')
    )
    xs = df2['x']['sum'].values
    ns = df2['x']['count'].values
    hs = df2.h.values

    trailing_i = len(df2) - 1
    x_sum = 0
    n_sum = 0
    # We don't know output size but it is bounded by len(df2)
    res_x = np.empty(len(df2))

    # Start from the right and work backwards
    for i in range(len(df2) - 1, -1, -1):
        x_sum += xs[i]
        n_sum += ns[i]
        while n_sum >= w:
            # Include points from the previous horizon. All of them if still
            # less than w, otherwise weight the mean by the difference
            excess_n = n_sum - w
            excess_x = excess_n * xs[i] / ns[i]
            res_x[trailing_i] = (x_sum - excess_x)/ w
            x_sum -= xs[trailing_i]
            n_sum -= ns[trailing_i]
            trailing_i -= 1

    res_h = hs[(trailing_i + 1):]
    res_x = res_x[(trailing_i + 1):]

    return pd.DataFrame({'horizon': res_h, name: res_x})



def rolling_median_by_h(x: np.ndarray, h: np.ndarray, w: int, name: str) -> pd.DataFrame:
    """Compute a rolling median of x, after first aggregating by h.

    Right-aligned. Computes a single median for each unique value of h. Each
    median is over at least w samples.

    For each h where there are fewer than w samples, we take samples from the previous h,
    moving backwards. (In other words, we ~ assume that the x's are shuffled within each h.)

    Parameters
    ----------
    x: Array.
    h: Array of horizon for each value in x.
    w: Integer window size (number of elements).
    name: Name for metric in result dataframe

    Returns
    -------
    Dataframe with columns horizon and name, the rolling median of x.
    """
    # Aggregate over h
    df = pd.DataFrame({'x': x, 'h': h})
    grouped = df.groupby('h')
    df2 = grouped.size().reset_index().sort_values('h')
    hs = df2['h']

    res_h = []
    res_x = []
    # Start from the right and work backwards
    i = len(hs) - 1
    while i >= 0:
        h_i = hs[i]
        xs = grouped.get_group(h_i).x.tolist()

        # wrap in array so this works if h is pandas Series with custom index or numpy array
        next_idx_to_add = np.array(h == h_i).argmax() - 1
        while (len(xs) < w) and (next_idx_to_add >= 0):
            # Include points from the previous horizon. All of them if still
            # less than w, otherwise just enough to get to w.
            xs.append(x[next_idx_to_add])
            next_idx_to_add -= 1
        if len(xs) < w:
            # Ran out of points before getting enough.
            break
        res_h.append(hs[i])
        res_x.append(np.median(xs))
        i -= 1
    res_h.reverse()
    res_x.reverse()
    return pd.DataFrame({'horizon': res_h, name: res_x})


# The functions below specify performance metrics for cross-validation results.
# Each takes as input the output of cross_validation, and returns the statistic
# as a dataframe, given a window size for rolling aggregation.


@register_performance_metric
def mse(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Mean squared error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mse.
    """
    se = (df['y'] - df['yhat']) ** 2
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mse': se})
    return rolling_mean_by_h(
        x=cast(np.ndarray, se.values),
        h=cast(np.ndarray, df['horizon'].values),
        w=w,
        name='mse',
    )


@register_performance_metric
def rmse(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Root mean squared error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and rmse.
    """
    res = mse(df, w)
    res['mse'] = np.sqrt(res['mse'])
    res.rename({'mse': 'rmse'}, axis='columns', inplace=True)
    return res


@register_performance_metric
def mae(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Mean absolute error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mae.
    """
    ae = (df['y'] - df['yhat']).abs()
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mae': ae})
    return rolling_mean_by_h(
        x=cast(np.ndarray, ae.values),
        h=cast(np.ndarray, df['horizon'].values),
        w=w,
        name='mae',
    )


@register_performance_metric
def mape(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Mean absolute percent error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mape.
    """
    ape = ((df['y'] - df['yhat']) / df['y']).abs()
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mape': ape})
    return rolling_mean_by_h(
        x=cast(np.ndarray, ape.values),
        h=cast(np.ndarray, df['horizon'].values),
        w=w,
        name='mape',
    )


@register_performance_metric
def mdape(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Median absolute percent error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mdape.
    """
    ape = ((df['y'] - df['yhat']) / df['y']).abs()
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mdape': ape})
    return rolling_median_by_h(
        x=cast(np.ndarray, ape.values),
        h=cast(np.ndarray, df['horizon'].values),
        w=w,
        name='mdape',
    )


@register_performance_metric
def smape(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Symmetric mean absolute percentage error
    based on Chen and Yang (2004) formula

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and smape.
    """
    sape = (df['y'] - df['yhat']).abs() / ((np.abs(df['y']) + np.abs(df['yhat'])) / 2)
    sape = sape.fillna(0)
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'smape': sape})
    return rolling_mean_by_h(
        x=cast(np.ndarray, sape.values),
        h=cast(np.ndarray, df['horizon'].values),
        w=w,
        name='smape',
    )


@register_performance_metric
def coverage(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """Coverage

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and coverage.
    """
    is_covered = (df['y'] >= df['yhat_lower']) & (df['y'] <= df['yhat_upper'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'coverage': is_covered})
    return rolling_mean_by_h(
        x=cast(np.ndarray, is_covered.values),
        h=cast(np.ndarray, df['horizon'].values),
        w=w,
        name='coverage',
    )


__all__ = [
    "PERFORMANCE_METRICS",
    "register_performance_metric",
    "performance_metrics",
    "rolling_mean_by_h",
    "rolling_median_by_h",
    "mse",
    "rmse",
    "mae",
    "mape",
    "mdape",
    "smape",
    "coverage",
]
