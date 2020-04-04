# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import json

import numpy as np
import pandas as pd

from fbprophet.forecaster import Prophet
from fbprophet import __version__


SIMPLE_ATTRIBUTES = [
    'growth', 'n_changepoints', 'specified_changepoints', 'changepoint_range',
    'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
    'seasonality_mode', 'seasonality_prior_scale', 'changepoint_prior_scale',
    'holidays_prior_scale', 'mcmc_samples', 'interval_width', 'uncertainty_samples',
    'y_scale', 'logistic_floor', 'country_holidays', 'component_modes', 'fit_kwargs'
]

PD_SERIES = ['changepoints', 'history_dates', 'train_holiday_names']

PD_TIMESTAMP = ['start']

PD_TIMEDELTA = ['t_scale']

PD_DATAFRAME = ['holidays', 'history', 'train_component_cols']

NP_ARRAY = ['changepoints_t']

ORDEREDDICT = ['seasonalities', 'extra_regressors']


def model_to_json(model):
    """Serialize a Prophet model to json string.

    Model must be fitted. Skips Stan objects that are not needed for predict.

    Can be deserialized with model_from_json.

    Parameters
    ----------
    model: Prophet model object.

    Returns
    -------
    json string that can be deserialized into a Prophet model.
    """
    if model.history is None:
        raise ValueError(
            "This can only be used to serialize models that have already been fit."
        )

    model_json = {
        attribute: getattr(model, attribute) for attribute in SIMPLE_ATTRIBUTES
    }
    # Handle attributes of non-core types
    for attribute in PD_SERIES:
        if getattr(model, attribute) is None:
            model_json[attribute] = None
        else:
            model_json[attribute] = getattr(model, attribute).to_json(
                orient='split', date_format='iso'
            )
    for attribute in PD_TIMESTAMP:
        model_json[attribute] = getattr(model, attribute).timestamp()
    for attribute in PD_TIMEDELTA:
        model_json[attribute] = getattr(model, attribute).total_seconds()
    for attribute in PD_DATAFRAME:
        if getattr(model, attribute) is None:
            model_json[attribute] = None
        else:
            model_json[attribute] = getattr(model, attribute).to_json(orient='table', index=False)
    for attribute in NP_ARRAY:
        model_json[attribute] = getattr(model, attribute).tolist()
    for attribute in ORDEREDDICT:
        model_json[attribute] = [
            list(getattr(model, attribute).keys()),
            getattr(model, attribute),
        ]
    # Params (Dict[str, np.ndarray])
    model_json['params'] = {k: v.tolist() for k, v in model.params.items()}
    # Attributes that are skipped: stan_fit, stan_backend
    model_json['__fbprophet_version'] = __version__
    return json.dumps(model_json)


def model_from_json(model_json):
    """Deserialize a Prophet model from json string.

    Deserializes models that were serialized with model_to_json.

    Parameters
    ----------
    model_json: Serialized model string

    Returns
    -------
    Prophet model.
    """
    attr_dict = json.loads(model_json)
    model = Prophet()  # We will overwrite all attributes set in init anyway
    # Simple types
    for attribute in SIMPLE_ATTRIBUTES:
        setattr(model, attribute, attr_dict[attribute])
    for attribute in PD_SERIES:
        if attr_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            s = pd.read_json(attr_dict[attribute], typ='series', orient='split')
            if s.name == 'ds':
                if len(s) == 0:
                    s = pd.to_datetime(s)
                s = s.dt.tz_localize(None)
            setattr(model, attribute, s)
    for attribute in PD_TIMESTAMP:
        setattr(model, attribute, pd.Timestamp.utcfromtimestamp(attr_dict[attribute]))
    for attribute in PD_TIMEDELTA:
        setattr(model, attribute, pd.Timedelta(seconds=attr_dict[attribute]))
    for attribute in PD_DATAFRAME:
        if attr_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            df = pd.read_json(attr_dict[attribute], typ='frame', orient='table', convert_dates=['ds'])
            if attribute == 'train_component_cols':
                # Special handling because of named index column
                df.columns.name = 'component'
                df.index.name = 'col'
            setattr(model, attribute, df)
    for attribute in NP_ARRAY:
        setattr(model, attribute, np.array(attr_dict[attribute]))
    for attribute in ORDEREDDICT:
        key_list, unordered_dict = attr_dict[attribute]
        od = OrderedDict()
        for key in key_list:
            od[key] = unordered_dict[key]
        setattr(model, attribute, od)
    # Params (Dict[str, np.ndarray])
    model.params = {k: np.array(v) for k, v in attr_dict['params'].items()}
    # Skipped attributes
    model.stan_backend = None
    model.stan_fit = None
    return model
