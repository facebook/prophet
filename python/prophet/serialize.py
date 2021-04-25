# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from copy import deepcopy
import json

import numpy as np
import pandas as pd

from prophet.forecaster import Prophet
from prophet import __version__


SIMPLE_ATTRIBUTES = [
    'growth', 'n_changepoints', 'specified_changepoints', 'changepoint_range',
    'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
    'seasonality_mode', 'seasonality_prior_scale', 'changepoint_prior_scale',
    'holidays_prior_scale', 'mcmc_samples', 'interval_width', 'uncertainty_samples',
    'y_scale', 'logistic_floor', 'country_holidays', 'component_modes'
]

PD_SERIES = ['changepoints', 'history_dates', 'train_holiday_names']

PD_TIMESTAMP = ['start']

PD_TIMEDELTA = ['t_scale']

PD_DATAFRAME = ['holidays', 'history', 'train_component_cols']

NP_ARRAY = ['changepoints_t']

ORDEREDDICT = ['seasonalities', 'extra_regressors']


def model_to_dict(model):
    """Convert a Prophet model to a dictionary suitable for JSON serialization.

    Model must be fitted. Skips Stan objects that are not needed for predict.

    Can be reversed with model_from_dict.

    Parameters
    ----------
    model: Prophet model object.

    Returns
    -------
    dict that can be used to serialize a Prophet model as JSON or loaded back
    into a Prophet model.
    """
    if model.history is None:
        raise ValueError(
            "This can only be used to serialize models that have already been fit."
        )

    model_dict = {
        attribute: getattr(model, attribute) for attribute in SIMPLE_ATTRIBUTES
    }
    # Handle attributes of non-core types
    for attribute in PD_SERIES:
        if getattr(model, attribute) is None:
            model_dict[attribute] = None
        else:
            model_dict[attribute] = getattr(model, attribute).to_json(
                orient='split', date_format='iso'
            )
    for attribute in PD_TIMESTAMP:
        model_dict[attribute] = getattr(model, attribute).timestamp()
    for attribute in PD_TIMEDELTA:
        model_dict[attribute] = getattr(model, attribute).total_seconds()
    for attribute in PD_DATAFRAME:
        if getattr(model, attribute) is None:
            model_dict[attribute] = None
        else:
            model_dict[attribute] = getattr(model, attribute).to_json(orient='table', index=False)
    for attribute in NP_ARRAY:
        model_dict[attribute] = getattr(model, attribute).tolist()
    for attribute in ORDEREDDICT:
        model_dict[attribute] = [
            list(getattr(model, attribute).keys()),
            getattr(model, attribute),
        ]
    # Other attributes with special handling
    # fit_kwargs -> Transform any numpy types before serializing.
    # They do not need to be transformed back on deserializing.
    fit_kwargs = deepcopy(model.fit_kwargs)
    if 'init' in fit_kwargs:
        for k, v in fit_kwargs['init'].items():
            if isinstance(v, np.ndarray):
                fit_kwargs['init'][k] = v.tolist()
            elif isinstance(v, np.floating):
                fit_kwargs['init'][k] = float(v)
    model_dict['fit_kwargs'] = fit_kwargs

    # Params (Dict[str, np.ndarray])
    model_dict['params'] = {k: v.tolist() for k, v in model.params.items()}
    # Attributes that are skipped: stan_fit, stan_backend
    model_dict['__prophet_version'] = __version__
    return model_dict


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
    model_json = model_to_dict(model)
    return json.dumps(model_json)


def model_from_dict(model_dict):
    """Recreate a Prophet model from a dictionary.

    Recreates models that were converted with model_to_dict.

    Parameters
    ----------
    model_dict: Dictionary containing model, created with model_to_dict.

    Returns
    -------
    Prophet model.
    """
    model = Prophet()  # We will overwrite all attributes set in init anyway
    # Simple types
    for attribute in SIMPLE_ATTRIBUTES:
        setattr(model, attribute, model_dict[attribute])
    for attribute in PD_SERIES:
        if model_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            s = pd.read_json(model_dict[attribute], typ='series', orient='split')
            if s.name == 'ds':
                if len(s) == 0:
                    s = pd.to_datetime(s)
                s = s.dt.tz_localize(None)
            setattr(model, attribute, s)
    for attribute in PD_TIMESTAMP:
        setattr(model, attribute, pd.Timestamp.utcfromtimestamp(model_dict[attribute]))
    for attribute in PD_TIMEDELTA:
        setattr(model, attribute, pd.Timedelta(seconds=model_dict[attribute]))
    for attribute in PD_DATAFRAME:
        if model_dict[attribute] is None:
            setattr(model, attribute, None)
        else:
            df = pd.read_json(model_dict[attribute], typ='frame', orient='table', convert_dates=['ds'])
            if attribute == 'train_component_cols':
                # Special handling because of named index column
                df.columns.name = 'component'
                df.index.name = 'col'
            setattr(model, attribute, df)
    for attribute in NP_ARRAY:
        setattr(model, attribute, np.array(model_dict[attribute]))
    for attribute in ORDEREDDICT:
        key_list, unordered_dict = model_dict[attribute]
        od = OrderedDict()
        for key in key_list:
            od[key] = unordered_dict[key]
        setattr(model, attribute, od)
    # Other attributes with special handling
    # fit_kwargs
    model.fit_kwargs = model_dict['fit_kwargs']
    # Params (Dict[str, np.ndarray])
    model.params = {k: np.array(v) for k, v in model_dict['params'].items()}
    # Skipped attributes
    model.stan_backend = None
    model.stan_fit = None
    return model


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
    model_dict = json.loads(model_json)
    return model_from_dict(model_dict)
