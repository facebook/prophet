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
