# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy
from functools import reduce
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_cutoffs(df, horizon, initial, period):
    """Generate cutoff dates

    Parameters
    ----------
    df: pd.DataFrame with historical data.
    horizon: pd.Timedelta forecast horizon.
    initial: pd.Timedelta window of the initial forecast period.
    period: pd.Timedelta simulated forecasts are done with this period.

    Returns
    -------
    list of pd.Timestamp
    """
    # Last cutoff is 'latest date in data - horizon' date
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]
    while result[-1] >= min(df['ds']) + initial:
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
            # Next cutoff point is 'last date before cutoff in data - horizon'
            closest_date = df[df['ds'] <= cutoff].max()['ds']
            cutoff = closest_date - horizon
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            'Less data than horizon after initial window. '
            'Make horizon or initial shorter.'
        )
    logger.info('Making {} forecasts with cutoffs between {} and {}'.format(
        len(result), result[-1], result[0]
    ))
    return reversed(result)


def cross_validation(model, horizon, period=None, initial=None):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points. Beginning from
    (end - horizon), works backwards making cutoffs with a spacing of period
    until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will begin here. If not provided, 3 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    df = model.history.copy().reset_index(drop=True)
    te = df['ds'].max()
    ts = df['ds'].min()
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    initial = 3 * horizon if initial is None else pd.Timedelta(initial)

    cutoffs = generate_cutoffs(df, horizon, initial, period)
    predicts = []
    for cutoff in cutoffs:
        # Generate new object with copying fitting options
        m = prophet_copy(model, cutoff)
        # Train model
        history_c = df[df['ds'] <= cutoff]
        if history_c.shape[0] < 2:
            raise Exception(
                'Less than two datapoints before cutoff. '
                'Increase initial window.'
            )
        m.fit(history_c)
        # Calculate yhat
        index_predicted = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
        # Get the columns for the future dataframe
        columns = ['ds']
        if m.growth == 'logistic':
            columns.append('cap')
            if m.logistic_floor:
                columns.append('floor')
        columns.extend(m.extra_regressors.keys())
        yhat = m.predict(df[index_predicted][columns])
        # Merge yhat(predicts), y(df, original data) and cutoff
        predicts.append(pd.concat([
            yhat[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            df[index_predicted][['y']].reset_index(drop=True),
            pd.DataFrame({'cutoff': [cutoff] * len(yhat)})
        ], axis=1))

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return reduce(lambda x, y: x.append(y), predicts).reset_index(drop=True)

def prophet_copy(m, cutoff=None):
    """Copy Prophet object

    Parameters
    ----------
    m: Prophet model.
    cutoff: pd.Timestamp or None, default None.
        cuttoff Timestamp for changepoints member variable.
        changepoints are only retained if 'changepoints <= cutoff'

    Returns
    -------
    Prophet class object with the same parameter with model variable
    """
    if m.history is None:
        raise Exception('This is for copying a fitted Prophet object.')

    if m.specified_changepoints:
        changepoints = m.changepoints
        if cutoff is not None:
            # Filter change points '<= cutoff'
            changepoints = changepoints[changepoints <= cutoff]
    else:
        changepoints = None

    # Auto seasonalities are set to False because they are already set in
    # m.seasonalities.
    m2 = m.__class__(
        growth=m.growth,
        n_changepoints=m.n_changepoints,
        changepoint_range=m.changepoint_range,
        changepoints=changepoints,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=m.holidays,
        seasonality_mode=m.seasonality_mode,
        seasonality_prior_scale=m.seasonality_prior_scale,
        changepoint_prior_scale=m.changepoint_prior_scale,
        holidays_prior_scale=m.holidays_prior_scale,
        mcmc_samples=m.mcmc_samples,
        interval_width=m.interval_width,
        uncertainty_samples=m.uncertainty_samples,
    )
    m2.extra_regressors = deepcopy(m.extra_regressors)
    m2.seasonalities = deepcopy(m.seasonalities)
    return m2


def performance_metrics(df, metrics=None, rolling_window=0.1):
    """Compute performance metrics from cross-validation results.

    Computes a suite of performance metrics on the output of cross-validation.
    By default the following metrics are included:
    'mse': mean squared error
    'rmse': root mean squared error
    'mae': mean absolute error
    'mape': mean percent error
    'coverage': coverage of the upper and lower intervals

    A subset of these can be specified by passing a list of names as the
    `metrics` argument.

    Metrics are calculated over a rolling window of cross validation
    predictions, after sorting by horizon. The size of that window (number of
    simulated forecast points) is determined by the rolling_window argument,
    which specifies a proportion of simulated forecast points to include in
    each window. rolling_window=0 will compute it separately for each simulated
    forecast point (i.e., 'mse' will actually be squared error with no mean).
    The default of rolling_window=0.1 will use 10% of the rows in df in each
    window. rolling_window=1 will compute the metric across all simulated forecast
    points. The results are set to the right edge of the window.

    The output is a dataframe containing column 'horizon' along with columns
    for each of the metrics computed.

    Parameters
    ----------
    df: The dataframe returned by cross_validation.
    metrics: A list of performance metrics to compute. If not provided, will
        use ['mse', 'rmse', 'mae', 'mape', 'coverage'].
    rolling_window: Proportion of data to use in each rolling window for
        computing the metrics. Should be in [0, 1].

    Returns
    -------
    Dataframe with a column for each metric, and column 'horizon'
    """
    valid_metrics = ['mse', 'rmse', 'mae', 'mape', 'coverage']
    if metrics is None:
        metrics = valid_metrics
    if len(set(metrics)) != len(metrics):
        raise ValueError('Input metrics must be a list of unique values')
    if not set(metrics).issubset(set(valid_metrics)):
        raise ValueError(
            'Valid values for metrics are: {}'.format(valid_metrics)
        )
    df_m = df.copy()
    df_m['horizon'] = df_m['ds'] - df_m['cutoff']
    df_m.sort_values('horizon', inplace=True)
    # Window size
    w = int(rolling_window * df_m.shape[0])
    w = max(w, 1)
    w = min(w, df_m.shape[0])
    cols = ['horizon']
    for metric in metrics:
        df_m[metric] = eval(metric)(df_m, w)
        cols.append(metric)
    df_m = df_m[cols]
    return df_m.dropna()


def rolling_mean(x, w):
    """Compute a rolling mean of x

    Right-aligned. Padded with NaNs on the front so the output is the same
    size as x.

    Parameters
    ----------
    x: Array.
    w: Integer window size (number of elements).

    Returns
    -------
    Rolling mean of x with window size w.
    """
    s = np.cumsum(np.insert(x, 0, 0))
    prefix = np.empty(w - 1)
    prefix.fill(np.nan)
    return np.hstack((prefix, (s[w:] - s[:-w]) / float(w)))  # right-aligned


# The functions below specify performance metrics for cross-validation results.
# Each takes as input the output of cross_validation, and returns the statistic
# as an array, given a window size for rolling aggregation.


def mse(df, w):
    """Mean squared error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Array of mean squared errors.
    """
    se = (df['y'] - df['yhat']) ** 2
    return rolling_mean(se.values, w)


def rmse(df, w):
    """Root mean squared error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Array of root mean squared errors.
    """
    return np.sqrt(mse(df, w))


def mae(df, w):
    """Mean absolute error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Array of mean absolute errors.
    """
    ae = np.abs(df['y'] - df['yhat'])
    return rolling_mean(ae.values, w)


def mape(df, w):
    """Mean absolute percent error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Array of mean absolute percent errors.
    """
    ape = np.abs((df['y'] - df['yhat']) / df['y'])
    return rolling_mean(ape.values, w)


def coverage(df, w):
    """Coverage

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Array of coverages.
    """
    is_covered = (df['y'] >= df['yhat_lower']) & (df['y'] <= df['yhat_upper'])
    return rolling_mean(is_covered.values, w)
