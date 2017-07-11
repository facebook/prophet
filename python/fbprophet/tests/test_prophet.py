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

DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data.csv'),
    parse_dates=['ds'],
)
DATA2 = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data2.csv'),
    parse_dates=['ds'],
)
# fb-block 1 end
# fb-block 2


class TestProphet(TestCase):

    def test_fit_predict(self):
        N = DATA.shape[0]
        train = DATA.head(N // 2)
        future = DATA.tail(N // 2)

        forecaster = Prophet()
        forecaster.fit(train)
        forecaster.predict(future)

    def test_fit_predict_no_seasons(self):
        N = DATA.shape[0]
        train = DATA.head(N // 2)
        future = DATA.tail(N // 2)

        forecaster = Prophet(weekly_seasonality=False, yearly_seasonality=False)
        forecaster.fit(train)
        forecaster.predict(future)

    def test_fit_predict_no_changepoints(self):
        N = DATA.shape[0]
        train = DATA.head(N // 2)
        future = DATA.tail(N // 2)

        forecaster = Prophet(n_changepoints=0)
        forecaster.fit(train)
        forecaster.predict(future)

    def test_fit_changepoint_not_in_history(self):
        train = DATA[(DATA['ds'] < '2013-01-01') | (DATA['ds'] > '2014-01-01')]
        train[(train['ds'] > '2014-01-01')] += 20
        future = pd.DataFrame({'ds': DATA['ds']})
        forecaster = Prophet(changepoints=['2013-06-06'])
        forecaster.fit(train)
        forecaster.predict(future)

    def test_fit_predict_duplicates(self):
        N = DATA.shape[0]
        train1 = DATA.head(N // 2).copy()
        train2 = DATA.head(N // 2).copy()
        train2['y'] += 10
        train = train1.append(train2)
        future = pd.DataFrame({'ds': DATA['ds'].tail(N // 2)})
        forecaster = Prophet()
        forecaster.fit(train)
        forecaster.predict(future)

    def test_setup_dataframe(self):
        m = Prophet()
        N = DATA.shape[0]
        history = DATA.head(N // 2).copy()

        history = m.setup_dataframe(history, initialize_scales=True)

        self.assertTrue('t' in history)
        self.assertEqual(history['t'].min(), 0.0)
        self.assertEqual(history['t'].max(), 1.0)

        self.assertTrue('y_scaled' in history)
        self.assertEqual(history['y_scaled'].max(), 1.0)

    def test_get_changepoints(self):
        m = Prophet()
        N = DATA.shape[0]
        history = DATA.head(N // 2).copy()

        history = m.setup_dataframe(history, initialize_scales=True)
        m.history = history

        m.set_changepoints()

        cp = m.changepoints_t
        self.assertEqual(cp.shape[0], m.n_changepoints)
        self.assertEqual(len(cp.shape), 1)
        self.assertTrue(cp.min() > 0)
        self.assertTrue(cp.max() < N)

        mat = m.get_changepoint_matrix()
        self.assertEqual(mat.shape[0], N // 2)
        self.assertEqual(mat.shape[1], m.n_changepoints)

    def test_get_zero_changepoints(self):
        m = Prophet(n_changepoints=0)
        N = DATA.shape[0]
        history = DATA.head(N // 2).copy()

        history = m.setup_dataframe(history, initialize_scales=True)
        m.history = history

        m.set_changepoints()
        cp = m.changepoints_t
        self.assertEqual(cp.shape[0], 1)
        self.assertEqual(cp[0], 0)

        mat = m.get_changepoint_matrix()
        self.assertEqual(mat.shape[0], N // 2)
        self.assertEqual(mat.shape[1], 1)

    def test_fourier_series_weekly(self):
        mat = Prophet.fourier_series(DATA['ds'], 7, 3)
        # These are from the R forecast package directly.
        true_values = np.array([
            0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837, -0.9009689,
        ])
        self.assertAlmostEqual(np.sum((mat[0] - true_values)**2), 0.0)

    def test_fourier_series_yearly(self):
        mat = Prophet.fourier_series(DATA['ds'], 365.25, 3)
        # These are from the R forecast package directly.
        true_values = np.array([
            0.7006152, -0.7135393, -0.9998330, 0.01827656, 0.7262249, 0.6874572,
        ])
        self.assertAlmostEqual(np.sum((mat[0] - true_values)**2), 0.0)

    def test_growth_init(self):
        model = Prophet(growth='logistic')
        history = DATA.iloc[:468].copy()
        history['cap'] = history['y'].max()

        history = model.setup_dataframe(history, initialize_scales=True)

        k, m = model.linear_growth_init(history)
        self.assertAlmostEqual(k, 0.3055671)
        self.assertAlmostEqual(m, 0.5307511)

        k, m = model.logistic_growth_init(history)

        self.assertAlmostEqual(k, 1.507925, places=4)
        self.assertAlmostEqual(m, -0.08167497, places=4)

    def test_piecewise_linear(self):
        model = Prophet()

        t = np.arange(11.)
        m = 0
        k = 1.0
        deltas = np.array([0.5])
        changepoint_ts = np.array([5])

        y = model.piecewise_linear(t, deltas, k, m, changepoint_ts)
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                           6.5, 8.0, 9.5, 11.0, 12.5])
        self.assertEqual((y - y_true).sum(), 0.0)

        t = t[8:]
        y_true = y_true[8:]
        y = model.piecewise_linear(t, deltas, k, m, changepoint_ts)
        self.assertEqual((y - y_true).sum(), 0.0)

    def test_piecewise_logistic(self):
        model = Prophet()

        t = np.arange(11.)
        cap = np.ones(11) * 10
        m = 0
        k = 1.0
        deltas = np.array([0.5])
        changepoint_ts = np.array([5])

        y = model.piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
        y_true = np.array([5.000000, 7.310586, 8.807971, 9.525741, 9.820138,
                           9.933071, 9.984988, 9.996646, 9.999252, 9.999833,
                           9.999963])
        self.assertAlmostEqual((y - y_true).sum(), 0.0, places=5)

        t = t[8:]
        y_true = y_true[8:]
        cap = cap[8:]
        y = model.piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
        self.assertAlmostEqual((y - y_true).sum(), 0.0, places=5)

    def test_holidays(self):
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2016-12-25']),
            'holiday': ['xmas'],
            'lower_window': [-1],
            'upper_window': [0],
        })
        model = Prophet(holidays=holidays)
        df = pd.DataFrame({
            'ds': pd.date_range('2016-12-20', '2016-12-31')
        })
        feats = model.make_holiday_features(df['ds'])
        # 11 columns generated even though only 8 overlap
        self.assertEqual(feats.shape, (df.shape[0], 2))
        self.assertEqual((feats.sum(0) - np.array([1.0, 1.0])).sum(), 0)

        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2016-12-25']),
            'holiday': ['xmas'],
            'lower_window': [-1],
            'upper_window': [10],
        })
        feats = Prophet(holidays=holidays).make_holiday_features(df['ds'])
        # 12 columns generated even though only 8 overlap
        self.assertEqual(feats.shape, (df.shape[0], 12))

    def test_fit_with_holidays(self):
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2012-06-06', '2013-06-06']),
            'holiday': ['seans-bday'] * 2,
            'lower_window': [0] * 2,
            'upper_window': [1] * 2,
        })
        model = Prophet(holidays=holidays, uncertainty_samples=0)
        model.fit(DATA).predict()

    def test_make_future_dataframe(self):
        N = 468
        train = DATA.head(N // 2)
        forecaster = Prophet()
        forecaster.fit(train)
        future = forecaster.make_future_dataframe(periods=3, freq='D',
                                                  include_history=False)
        correct = pd.DatetimeIndex(['2013-04-26', '2013-04-27', '2013-04-28'])
        self.assertEqual(len(future), 3)
        for i in range(3):
            self.assertEqual(future.iloc[i]['ds'], correct[i])

        future = forecaster.make_future_dataframe(periods=3, freq='M',
                                                  include_history=False)
        correct = pd.DatetimeIndex(['2013-04-30', '2013-05-31', '2013-06-30'])
        self.assertEqual(len(future), 3)
        for i in range(3):
            self.assertEqual(future.iloc[i]['ds'], correct[i])

    def test_auto_weekly_seasonality(self):
        # Should be enabled
        N = 15
        train = DATA.head(N)
        m = Prophet()
        self.assertEqual(m.weekly_seasonality, 'auto')
        m.fit(train)
        self.assertIn('weekly', m.seasonalities)
        self.assertEqual(m.seasonalities['weekly'], (7, 3))
        # Should be disabled due to too short history
        N = 9
        train = DATA.head(N)
        m = Prophet()
        m.fit(train)
        self.assertNotIn('weekly', m.seasonalities)
        m = Prophet(weekly_seasonality=True)
        m.fit(train)
        self.assertIn('weekly', m.seasonalities)
        # Should be False due to weekly spacing
        train = DATA.iloc[::7, :]
        m = Prophet()
        m.fit(train)
        self.assertNotIn('weekly', m.seasonalities)
        m = Prophet(weekly_seasonality=2)
        m.fit(DATA)
        self.assertEqual(m.seasonalities['weekly'], (7, 2))

    def test_auto_yearly_seasonality(self):
        # Should be enabled
        m = Prophet()
        self.assertEqual(m.yearly_seasonality, 'auto')
        m.fit(DATA)
        self.assertIn('yearly', m.seasonalities)
        self.assertEqual(m.seasonalities['yearly'], (365.25, 10))
        # Should be disabled due to too short history
        N = 240
        train = DATA.head(N)
        m = Prophet()
        m.fit(train)
        self.assertNotIn('yearly', m.seasonalities)
        m = Prophet(yearly_seasonality=True)
        m.fit(train)
        self.assertIn('yearly', m.seasonalities)
        m = Prophet(yearly_seasonality=7)
        m.fit(DATA)
        self.assertEqual(m.seasonalities['yearly'], (365.25, 7))

    def test_auto_daily_seasonality(self):
        # Should be enabled
        m = Prophet()
        self.assertEqual(m.daily_seasonality, 'auto')
        m.fit(DATA2)
        self.assertIn('daily', m.seasonalities)
        self.assertEqual(m.seasonalities['daily'], (1, 4))
        # Should be disabled due to too short history
        N = 430
        train = DATA2.head(N)
        m = Prophet()
        m.fit(train)
        self.assertNotIn('daily', m.seasonalities)
        m = Prophet(daily_seasonality=True)
        m.fit(train)
        self.assertIn('daily', m.seasonalities)
        m = Prophet(daily_seasonality=7)
        m.fit(DATA2)
        self.assertEqual(m.seasonalities['daily'], (1, 7))
        m = Prophet()
        m.fit(DATA)
        self.assertNotIn('daily', m.seasonalities)

    def test_subdaily_holidays(self):
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2017-01-02']),
            'holiday': ['special_day'],
        })
        m = Prophet(holidays=holidays)
        m.fit(DATA2)
        fcst = m.predict()
        self.assertEqual(sum(fcst['special_day'] == 0), 575)

    def test_custom_seasonality(self):
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2017-01-02']),
            'holiday': ['special_day'],
        })
        m = Prophet(holidays=holidays)
        m.add_seasonality(name='monthly', period=30, fourier_order=5)
        self.assertEqual(m.seasonalities['monthly'], (30, 5))
