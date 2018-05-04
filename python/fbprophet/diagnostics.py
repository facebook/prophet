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


def _cutoffs(df, horizon, k, period):
    """Generate cutoff dates

    Parameters
    ----------
    df: pd.DataFrame with historical data
    horizon: pd.Timedelta.
        Forecast horizon
    k: Int number.
        The number of forecasts point.
    period: pd.Timedelta.
        Simulated Forecast will be done at every this period.

    Returns
    -------
    list of pd.Timestamp
    """
    # Last cutoff is 'latest date in data - horizon' date
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]

    for i in range(1, k):
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
            # Next cutoff point is 'last date before cutoff in data - horizon'
            closest_date = df[df['ds'] <= cutoff].max()['ds']
            cutoff = closest_date - horizon
        if cutoff < df['ds'].min():
            logger.warning(
                'Not enough data for requested number of cutoffs! '
                'Using {}.'.format(i))
            break
        result.append(cutoff)

    # Sort lines in ascending order
    return reversed(result)


def simulated_historical_forecasts(model, horizon, k, period=None):
    """Simulated Historical Forecasts.

    Make forecasts from k historical cutoff points, working backwards from
    (end - horizon) with a spacing of period between each cutoff.

    Parameters
    ----------
    model: Prophet class object.
        Fitted Prophet model
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    k: Int number of forecasts point.
    period: Optional string with pd.Timedelta compatible style. Simulated
        forecast will be done at every this period. If not provided,
        0.5 * horizon is used.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    cutoffs = _cutoffs(df, horizon, k, period)
    predicts = []
    for cutoff in cutoffs:
        # Generate new object with copying fitting options
        m = prophet_copy(model, cutoff)
        # Train model
        m.fit(df[df['ds'] <= cutoff])
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


def cross_validation(model, horizon, period=None, initial=None):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points. Beginning from initial,
    makes cutoffs with a spacing of period up to (end - horizon).

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
    te = model.history['ds'].max()
    ts = model.history['ds'].min()
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    initial = 3 * horizon if initial is None else pd.Timedelta(initial)
    k = int(np.ceil(((te - horizon) - (ts + initial)) / period))
    if k < 1:
        raise ValueError(
            'Not enough data for specified horizon, period, and initial.')
    return simulated_historical_forecasts(model, horizon, k, period)


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
        changepoints=changepoints,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=m.holidays,
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


def performance_metrics(df, metrics=None, aggregation='horizon'):
    """Compute performance metrics from cross-validation results.

    Computes a suite of performance metrics on the output of cross-validation.
    By default the following metrics are included:
    'mse': mean squared error
    'mae': mean absolute error
    'mape': mean percent error
    'coverage': coverage of the upper and lower intervals

    A subset of these can be specified by passing a list of names as the
    `metrics` argument.

    By default, metrics will be computed for each horizon (ds - cutoff).
    Alternatively, metrics can be computed at the level of individual ds/cutoff
    pairs (aggregation='none'), or aggregated over all ds/cutoffs
    (aggregation='all').

    The output is a dataframe containing the columns corresponding to the level
    of aggregation ('horizon', 'ds' and 'cutoff', or none) along with columns
    for each of the metrics computed.

    Parameters
    ----------
    df: The dataframe returned by cross_validation.
    metrics: A list of performance metrics to compute. If not provided, will
        use ['mse', 'mae', 'mape', 'coverage'].
    aggregation: Level of aggregation for computing performance statistics.
        Must be 'horizon', 'none', or 'all'.

    Returns
    -------
    Dataframe with a column for each metric, and a combination of columns 'ds',
    'cutoff', and 'horizon', depending on the aggregation level.
    """
    # Input validation
    valid_aggregations = ['horizon', 'all', 'none']
    if aggregation not in valid_aggregations:
        raise ValueError(
            'Aggregation {} is not valid; must be one of {}'.format(
                aggregation, valid_agggregations
            )
        )
    valid_metrics = ['mse', 'mae', 'mape', 'coverage']
    if metrics is None:
        metrics = valid_metrics
    if len(set(metrics)) != len(metrics):
        raise ValueError('Input metrics must be a list of unique values')
    if not set(metrics).issubset(set(valid_metrics)):
        raise ValueError(
            'Valid values for metrics are: {}'.format(valid_metrics)
        )
    # Get function for the metrics we want
    metric_fns = {m: eval(m) for m in metrics}
    def all_metrics(df_g):
        return pd.Series({name: fn(df_g) for name, fn in metric_fns.items()})
    # Apply functions to groupby
    if aggregation == 'all':
        return all_metrics(df)
    # else,
    df_m = df.copy()
    df_m['horizon'] = df_m['ds'] - df_m['cutoff']
    if aggregation == 'horizon':
        return df_m.groupby('horizon').apply(all_metrics).reset_index()
    # else,
    for name, fn in metric_fns.items():
        df_m[name] = fn(df_m, agg=False)
    return df_m


# The functions below specify performance metrics for cross-validation results.
# Each takes as input the output of cross_validation, and has two modes of
# return: if agg=True, returns a float that is the metric aggregated over the
# input. If agg=False, returns results without aggregation (for
# aggregation='none' in performance_metrics).


def mse(df, agg=True):
    """Mean squared error
    """
    se = (df['y'] - df['yhat']) ** 2
    if agg:
        return np.mean(se)
    return se


def mae(df, agg=True):
    """Mean absolute error
    """
    ae = np.abs(df['y'] - df['yhat'])
    if agg:
        return np.mean(ae)
    return ae


def mape(df, agg=True):
    """Mean absolute percent error
    """
    ape = np.abs((df['y'] - df['yhat']) / df['y'])
    if agg:
        return np.mean(ape)
    return ape


def coverage(df, agg=True):
    """Coverage
    """
    is_covered = (df['y'] >= df['yhat_lower']) & (df['y'] <= df['yhat_upper'])
    if agg:
        return np.mean(is_covered)
    return is_covered
