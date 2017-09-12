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

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from functools import reduce


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
        m = model.copy(cutoff)
        # Train model
        m.fit(df[df['ds'] <= cutoff])
        # Calculate yhat
        index_predicted = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
        columns = ['ds']
        if m.growth == 'logistic':
            columns.append('cap')
            if m.logistic_floor:
                columns.append('floor')
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
