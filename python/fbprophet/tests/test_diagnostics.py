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

import os
import numpy as np
import pandas as pd

from unittest import TestCase
from fbprophet import Prophet
from fbprophet import diagnostics


class TestDiagnostics(TestCase):

    def test_cv(self):
        # Use frist 100 record in data.csv
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'), parse_dates = ['ds']).head(100)
        m = Prophet()
        m.fit(df)
        for periods in [5, 10]:
            for horizon in [1, 3]:
                df_result = diagnostics.cv(m, periods=periods, horizon=horizon)
                # The size of output should be equal to 'periods'
                self.assertEqual(len(df_result), periods)
                # All data should be equal
                self.assertTrue(np.all(df_result.y == df.tail(periods).reset_index(drop=True).y))

