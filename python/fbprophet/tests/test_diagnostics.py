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

import numpy as np
import pandas as pd

# fb-block 1 start
import os
from unittest import TestCase
from fbprophet import Prophet
from fbprophet import diagnostics

DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data.csv'), parse_dates=['ds']
).head(100)
# fb-block 1 end
# fb-block 2


class TestDiagnostics(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDiagnostics, self).__init__(*args, **kwargs)
        # Use first 100 record in data.csv
        self.__df = DATA

    def test_simulated_historical_forecasts(self):
        m = Prophet()
        m.fit(self.__df)
        k = 2
        for p in [1, 10]:
            for h in [1, 3]:
                period = '{} days'.format(p)
                horizon = '{} days'.format(h)
                df_shf = diagnostics.simulated_historical_forecasts(
                    m, horizon=horizon, k=k, period=period)
                # All cutoff dates should be less than ds dates
                self.assertTrue((df_shf['cutoff'] < df_shf['ds']).all())
                # The unique size of output cutoff should be equal to 'k'
                self.assertEqual(len(np.unique(df_shf['cutoff'])), k)
                self.assertEqual(
                    max(df_shf['ds'] - df_shf['cutoff']),
                    pd.Timedelta(horizon),
                )
                dc = df_shf['cutoff'].diff()
                dc = dc[dc > pd.Timedelta(0)].min()
                self.assertTrue(dc >= pd.Timedelta(period))
                # Each y in df_shf and self.__df with same ds should be equal
                df_merged = pd.merge(df_shf, self.__df, 'left', on='ds')
                self.assertAlmostEqual(
                    np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 0.0)

    def test_simulated_historical_forecasts_logistic(self):
        m = Prophet(growth='logistic')
        df = self.__df.copy()
        df['cap'] = 40
        m.fit(df)
        df_shf = diagnostics.simulated_historical_forecasts(
            m, horizon='3 days', k=2, period='3 days')
        # All cutoff dates should be less than ds dates
        self.assertTrue((df_shf['cutoff'] < df_shf['ds']).all())
        # The unique size of output cutoff should be equal to 'k'
        self.assertEqual(len(np.unique(df_shf['cutoff'])), 2)
        # Each y in df_shf and self.__df with same ds should be equal
        df_merged = pd.merge(df_shf, df, 'left', on='ds')
        self.assertAlmostEqual(
            np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 0.0)

    def test_simulated_historical_forecasts_default_value_check(self):
        m = Prophet()
        m.fit(self.__df)
        # Default value of period should be equal to 0.5 * horizon
        df_shf1 = diagnostics.simulated_historical_forecasts(
            m, horizon='10 days', k=1)
        df_shf2 = diagnostics.simulated_historical_forecasts(
            m, horizon='10 days', k=1, period='5 days')
        self.assertAlmostEqual(
            ((df_shf1['y'] - df_shf2['y']) ** 2).sum(), 0.0)
        self.assertAlmostEqual(
            ((df_shf1['yhat'] - df_shf2['yhat']) ** 2).sum(), 0.0)

    def test_cross_validation(self):
        m = Prophet()
        m.fit(self.__df)
        # Calculate the number of cutoff points(k)
        horizon = pd.Timedelta('4 days')
        period = pd.Timedelta('10 days')
        k = 5
        df_cv = diagnostics.cross_validation(
            m, horizon='4 days', period='10 days', initial='90 days')
        # The unique size of output cutoff should be equal to 'k'
        self.assertEqual(len(np.unique(df_cv['cutoff'])), k)
        self.assertEqual(max(df_cv['ds'] - df_cv['cutoff']), horizon)
        dc = df_cv['cutoff'].diff()
        dc = dc[dc > pd.Timedelta(0)].min()
        self.assertTrue(dc >= period)

    def test_cross_validation_default_value_check(self):
        m = Prophet()
        m.fit(self.__df)
        # Default value of initial should be equal to 3 * horizon
        df_cv1 = diagnostics.cross_validation(
            m, horizon='32 days', period='10 days')
        df_cv2 = diagnostics.cross_validation(
            m, horizon='32 days', period='10 days', initial='96 days')
        self.assertAlmostEqual(
            ((df_cv1['y'] - df_cv2['y']) ** 2).sum(), 0.0)
        self.assertAlmostEqual(
            ((df_cv1['yhat'] - df_cv2['yhat']) ** 2).sum(), 0.0)
