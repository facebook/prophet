# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from unittest import TestCase

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.utilities import regressor_coefficients


DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data.csv'),
    parse_dates=['ds'],
)

class TestUtilities(TestCase):
    def test_regressor_coefficients(self):
        m = Prophet()
        N = DATA.shape[0]
        df = DATA.copy()
        np.random.seed(123)
        df['regr1'] = np.random.normal(size=N)
        df['regr2'] = np.random.normal(size=N)
        m.add_regressor('regr1', mode='additive')
        m.add_regressor('regr2', mode='multiplicative')
        m.fit(df)

        coefs = regressor_coefficients(m)
        self.assertTrue(coefs.shape == (2, 6))
        # No MCMC sampling, so lower and upper should be the same as mean
        self.assertTrue(np.array_equal(coefs['coef_lower'].values, coefs['coef'].values))
        self.assertTrue(np.array_equal(coefs['coef_upper'].values, coefs['coef'].values))
