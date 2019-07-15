# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import cmdstanpy
import pkg_resources


def get_prophet_stan_model() -> cmdstanpy.Model:
    """Load compiled Stan model"""
    model_file = pkg_resources.resource_filename(
        'fbprophet',
        'stan_model/prophet_model.bin',
    )
    return cmdstanpy.Model(exe_file=model_file)


prophet_stan_model: cmdstanpy.Model = get_prophet_stan_model()
