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

import pandas as pd
from functools import reduce

from fbprophet import Prophet


def cv(model, periods, horizon=1):
    """Computes the forecast errors obtained by applying predict function to subsets of
    the time series y using a rolling forecast origin.

    Parameters
    ----------
    model: Prophet class object
    periods: Int number of test periods of a rolling forecast origin.
    horizon: Forecast horizon

    Returns
    -------
    A pd.DataFrame with the forecast errors (error = y - yhat)
    """
    df = model.history.copy().reset_index(drop=True)
    size_history = len(df)
    predicts = []
    for i in range(periods):
        # Generate new object with copying fitting options
        model = Prophet(
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
        # Train model
        size_train = size_history - periods - horizon + 1 + i
        model.fit(df.head(size_train))
        # Calculate yhat
        df_future = pd.DataFrame({'ds': df.iloc[[size_train + horizon - 1]].ds}).reset_index(drop=True)
        predicts.append(model.predict(df_future))

    # Merge yhat(predicts) and y(df, original data)
    result = pd.concat([
        reduce(lambda x, y: x.append(y), predicts).reset_index(drop=True),
        df.tail(periods).reset_index(drop=True)['y']
    ], axis=1)
    result.reset_index(drop=True)
    result['error'] = result.y - result.yhat
    return result

