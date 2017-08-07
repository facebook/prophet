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
from fbprophet import Prophet

def _copy(model):
    result = Prophet(
        growth=model.growth,
        n_changepoints=model.n_changepoints,
        yearly_seasonality=model.yearly_seasonality,
        weekly_seasonality=model.weekly_seasonality,
        holidays=model.holidays,
        seasonality_prior_scale=model.seasonality_prior_scale,
        changepoint_prior_scale=model.changepoint_prior_scale,
        holidays_prior_scale=model.holidays_prior_scale,
        mcmc_samples=model.mcmc_samples,
        interval_width=model.interval_width,
        uncertainty_samples=model.uncertainty_samples
    )
    return result


def _cutoff_points(te, horizon, period, k):
    return [te - horizon - period * i for i in reversed(range(k))]


def shf(model, horizon, k, period=None):
    """Simulated Historical Forecasts.

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model
    horizon: string which has pd.Timedelta compatible style.
        Forecast horizon
    k: Int number. The number of forecasts point.
    period: string which has pd.Timedelta compatible style or None, default None
        Simulated Forecast will be done at every this period.
        0.5 * horizon is used when it is None.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    pd.to_timedelta
    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)
    period = 0.5 * horizon if period is None else pd.Timedelta(period)
    cutoffs = _cutoff_points(df['ds'].max(), horizon, period, k)

    predicts = []
    for cutoff in cutoffs:
        # Generate new object with copying fitting options
        m = _copy(model)
        # Train model
        m.fit(df[df['ds'] <= cutoff])
        # Calculate yhat
        index_predicted = (df['ds'] >= cutoff) & (df['ds'] <= cutoff + horizon)
        yhat = m.predict(df[index_predicted][['ds']])
        # Merge yhat(predicts), y(df, original data) and cutoff
        predicts.append(pd.concat([
            yhat,
            df[index_predicted][['y']].reset_index(drop=True),
            pd.DataFrame({'cutoff': [cutoff] * len(yhat)})
        ], axis=1))

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return reduce(lambda x, y: x.append(y), predicts).reset_index(drop=True)


def cv(model, horizon, period, initial=None):
    """Cross-Validation for time-series.

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model
    horizon: string which has pd.Timedelta compatible style.
        Forecast horizon
    period: string which has pd.Timedelta compatible style.
        Simulated Forecast will be done at every this period.
        This function is the same with Time series cross-validation described in https://robjhyndman.com/hyndsight/tscv/
        when the value of period is equal to the time interval of data.
    initial: string which has pd.Timedelta compatible style or None, default None
        First training period.
        3 * horizon is used when it is None.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    te = model.history['ds'].max()
    ts = model.history['ds'].min()
    horizon = pd.Timedelta(horizon)
    period = pd.Timedelta(period)
    initial = 3 * horizon if initial is None else pd.Timedelta(initial)
    k = int(np.floor(((te - horizon) - (ts + initial)) / period))
    return shf(model, horizon, k, period)
