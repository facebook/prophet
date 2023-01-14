# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

def regressor_index(m, name):
    """Given the name of a regressor, return its (column) index in the `beta` matrix.

    Parameters
    ----------
    m: Prophet model object, after fitting.
    name: Name of the regressor, as passed into the `add_regressor` function.

    Returns
    -------
    The column index of the regressor in the `beta` matrix.
    """
    return np.extract(
        m.train_component_cols[name] == 1, m.train_component_cols.index
    )[0]

def regressor_coefficients(m):
    """Summarise the coefficients of the extra regressors used in the model.

    For additive regressors, the coefficient represents the incremental impact
    on `y` of a unit increase in the regressor. For multiplicative regressors,
    the incremental impact is equal to `trend(t)` multiplied by the coefficient.

    Coefficients are measured on the original scale of the training data.

    Parameters
    ----------
    m: Prophet model object, after fitting.

    Returns
    -------
    pd.DataFrame containing:
    - `regressor`: Name of the regressor
    - `regressor_mode`: Whether the regressor has an additive or multiplicative
        effect on `y`.
    - `center`: The mean of the regressor if it was standardized. Otherwise 0.
    - `coef_lower`: Lower bound for the coefficient, estimated from the MCMC samples.
        Only different to `coef` if `mcmc_samples > 0`.
    - `coef`: Expected value of the coefficient.
    - `coef_upper`: Upper bound for the coefficient, estimated from MCMC samples.
        Only to different to `coef` if `mcmc_samples > 0`.
    """
    assert len(m.extra_regressors) > 0, 'No extra regressors found.'
    coefs = []
    for regressor, params in m.extra_regressors.items():
        beta = m.params['beta'][:, regressor_index(m, regressor)]
        if params['mode'] == 'additive':
            coef = beta * m.y_scale / params['std']
        else:
            coef = beta / params['std']
        percentiles = [
            (1 - m.interval_width) / 2,
            1 - (1 - m.interval_width) / 2,
        ]
        coef_bounds = np.quantile(coef, q=percentiles)
        record = {
            'regressor': regressor,
            'regressor_mode': params['mode'],
            'center': params['mu'],
            'coef_lower': coef_bounds[0],
            'coef': np.mean(coef),
            'coef_upper': coef_bounds[1],
        }
        coefs.append(record)

    return pd.DataFrame(coefs)

def warm_start_params(m):
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res
