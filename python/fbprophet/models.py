# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import pickle

import pkg_resources


def get_prophet_stan_model():
    """Load compiled Stan model"""
    model_file = pkg_resources.resource_filename(
        'fbprophet',
        'stan_model/prophet_model.pkl',
    )
    with open(model_file, 'rb') as f:
        return pickle.load(f)


prophet_stan_model = get_prophet_stan_model()
