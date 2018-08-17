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
from unittest import TestCase

import numpy as np
import pandas as pd

from fbprophet import Prophet

DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data.csv'),
    parse_dates=['ds'],
)
DATA2 = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data2.csv'),
    parse_dates=['ds'],
)


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

    def test_fit_predict_constant_history(self):
        N = DATA.shape[0]
        train = DATA.head(N // 2).copy()
        train['y'] = 20
        future = pd.DataFrame({'ds': DATA['ds'].tail(N // 2)})
        m = Prophet()
        m.fit(train)
        fcst = m.predict(future)
        self.assertEqual(fcst['yhat'].values[-1], 20)
        train['y'] = 0
        future = pd.DataFrame({'ds': DATA['ds'].tail(N // 2)})
        m = Prophet()
        m.fit(train)
        fcst = m.predict(future)
        self.assertEqual(fcst['yhat'].values[-1], 0)

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

    def test_logistic_floor(self):
        m = Prophet(growth='logistic')
        N = DATA.shape[0]
        history = DATA.head(N // 2).copy()
        history['floor'] = 10.
        history['cap'] = 80.
        future = DATA.tail(N // 2).copy()
        future['cap'] = 80.
        future['floor'] = 10.
        m.fit(history, algorithm='Newton')
        self.assertTrue(m.logistic_floor)
        self.assertTrue('floor' in m.history)
        self.assertAlmostEqual(m.history['y_scaled'][0], 1.)
        fcst1 = m.predict(future)

        m2 = Prophet(growth='logistic')
        history2 = history.copy()
        history2['y'] += 10.
        history2['floor'] += 10.
        history2['cap'] += 10.
        future['cap'] += 10.
        future['floor'] += 10.
        m2.fit(history2, algorithm='Newton')
        self.assertAlmostEqual(m2.history['y_scaled'][0], 1.)
        fcst2 = m2.predict(future)
        fcst2['yhat'] -= 10.
        # Check for approximate shift invariance
        self.assertTrue((np.abs(fcst1['yhat'] - fcst2['yhat']) < 1).all())

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
        cp_indx = int(np.ceil(0.8 * history.shape[0]))
        self.assertTrue(cp.max() <= history['t'].values[cp_indx])

    def test_set_changepoint_range(self):
        m = Prophet(changepoint_range=0.4)
        N = DATA.shape[0]
        history = DATA.head(N // 2).copy()

        history = m.setup_dataframe(history, initialize_scales=True)
        m.history = history

        m.set_changepoints()

        cp = m.changepoints_t
        self.assertEqual(cp.shape[0], m.n_changepoints)
        self.assertEqual(len(cp.shape), 1)
        self.assertTrue(cp.min() > 0)
        cp_indx = int(np.ceil(0.4 * history.shape[0]))
        self.assertTrue(cp.max() <= history['t'].values[cp_indx])
        with self.assertRaises(ValueError):
            m = Prophet(changepoint_range=-0.1)
        with self.assertRaises(ValueError):
            m = Prophet(changepoint_range=2)

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

    def test_override_n_changepoints(self):
        m = Prophet()
        history = DATA.head(20).copy()

        history = m.setup_dataframe(history, initialize_scales=True)
        m.history = history

        m.set_changepoints()
        self.assertEqual(m.n_changepoints, 15)
        cp = m.changepoints_t
        self.assertEqual(cp.shape[0], 15)

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
        feats, priors, names = model.make_holiday_features(df['ds'])
        # 11 columns generated even though only 8 overlap
        self.assertEqual(feats.shape, (df.shape[0], 2))
        self.assertEqual((feats.sum(0) - np.array([1.0, 1.0])).sum(), 0)
        self.assertEqual(priors, [10., 10.])  # Default prior
        self.assertEqual(names, ['xmas'])

        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2016-12-25']),
            'holiday': ['xmas'],
            'lower_window': [-1],
            'upper_window': [10],
        })
        m = Prophet(holidays=holidays)
        feats, priors, names = m.make_holiday_features(df['ds'])
        # 12 columns generated even though only 8 overlap
        self.assertEqual(feats.shape, (df.shape[0], 12))
        self.assertEqual(priors, list(10. * np.ones(12)))
        self.assertEqual(names, ['xmas'])
        # Check prior specifications
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2016-12-25', '2017-12-25']),
            'holiday': ['xmas', 'xmas'],
            'lower_window': [-1, -1],
            'upper_window': [0, 0],
            'prior_scale': [5., 5.],
        })
        m = Prophet(holidays=holidays)
        feats, priors, names = m.make_holiday_features(df['ds'])
        self.assertEqual(priors, [5., 5.])
        self.assertEqual(names, ['xmas'])
        # 2 different priors
        holidays2 = pd.DataFrame({
            'ds': pd.to_datetime(['2012-06-06', '2013-06-06']),
            'holiday': ['seans-bday'] * 2,
            'lower_window': [0] * 2,
            'upper_window': [1] * 2,
            'prior_scale': [8] * 2,
        })
        holidays2 = pd.concat((holidays, holidays2))
        m = Prophet(holidays=holidays2)
        feats, priors, names = m.make_holiday_features(df['ds'])
        pn = zip(priors, [s.split('_delim_')[0] for s in feats.columns])
        for t in pn:
            self.assertIn(t, [(8., 'seans-bday'), (5., 'xmas')])
        holidays2 = pd.DataFrame({
            'ds': pd.to_datetime(['2012-06-06', '2013-06-06']),
            'holiday': ['seans-bday'] * 2,
            'lower_window': [0] * 2,
            'upper_window': [1] * 2,
        })
        holidays2 = pd.concat((holidays, holidays2))
        feats, priors, names = Prophet(
            holidays=holidays2, holidays_prior_scale=4
        ).make_holiday_features(df['ds'])
        self.assertEqual(set(priors), {4., 5.})
        # Check incompatible priors
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2016-12-25', '2016-12-27']),
            'holiday': ['xmasish', 'xmasish'],
            'lower_window': [-1, -1],
            'upper_window': [0, 0],
            'prior_scale': [5., 6.],
        })
        with self.assertRaises(ValueError):
            Prophet(holidays=holidays).make_holiday_features(df['ds'])

    def test_fit_with_holidays(self):
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2012-06-06', '2013-06-06']),
            'holiday': ['seans-bday'] * 2,
            'lower_window': [0] * 2,
            'upper_window': [1] * 2,
        })
        model = Prophet(holidays=holidays, uncertainty_samples=0)
        model.fit(DATA).predict()

    def test_fit_predict_with_append_holidays(self):
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2012-06-06', '2013-06-06']),
            'holiday': ['seans-bday'] * 2,
            'lower_window': [0] * 2,
            'upper_window': [1] * 2,
        })
        append_holidays = 'US'
        # Test with holidays and append_holidays
        model = Prophet(holidays=holidays,
                        append_holidays=append_holidays,
                        uncertainty_samples=0)
        model.fit(DATA).predict()
        # There are training holidays missing in the test set
        train = DATA.head(154)
        future = DATA.tail(355)
        model = Prophet(append_holidays=append_holidays, uncertainty_samples=0)
        model.fit(train).predict(future)
        # There are test holidays missing in the training set
        train = DATA.tail(355)
        future = DATA2
        model = Prophet(append_holidays=append_holidays, uncertainty_samples=0)
        model.fit(train).predict(future)

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
        self.assertEqual(
            m.seasonalities['weekly'],
            {
                'period': 7,
                'fourier_order': 3,
                'prior_scale': 10.,
                'mode': 'additive',
            },
        )
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
        m = Prophet(weekly_seasonality=2, seasonality_prior_scale=3.)
        m.fit(DATA)
        self.assertEqual(
            m.seasonalities['weekly'],
            {
                'period': 7,
                'fourier_order': 2,
                'prior_scale': 3.,
                'mode': 'additive',
            },
        )

    def test_auto_yearly_seasonality(self):
        # Should be enabled
        m = Prophet()
        self.assertEqual(m.yearly_seasonality, 'auto')
        m.fit(DATA)
        self.assertIn('yearly', m.seasonalities)
        self.assertEqual(
            m.seasonalities['yearly'],
            {
                'period': 365.25,
                'fourier_order': 10,
                'prior_scale': 10.,
                'mode': 'additive',
            },
        )
        # Should be disabled due to too short history
        N = 240
        train = DATA.head(N)
        m = Prophet()
        m.fit(train)
        self.assertNotIn('yearly', m.seasonalities)
        m = Prophet(yearly_seasonality=True)
        m.fit(train)
        self.assertIn('yearly', m.seasonalities)
        m = Prophet(yearly_seasonality=7, seasonality_prior_scale=3.)
        m.fit(DATA)
        self.assertEqual(
            m.seasonalities['yearly'],
            {
                'period': 365.25,
                'fourier_order': 7,
                'prior_scale': 3.,
                'mode': 'additive',
            },
        )

    def test_auto_daily_seasonality(self):
        # Should be enabled
        m = Prophet()
        self.assertEqual(m.daily_seasonality, 'auto')
        m.fit(DATA2)
        self.assertIn('daily', m.seasonalities)
        self.assertEqual(
            m.seasonalities['daily'],
            {
                'period': 1,
                'fourier_order': 4,
                'prior_scale': 10.,
                'mode': 'additive',
            },
        )
        # Should be disabled due to too short history
        N = 430
        train = DATA2.head(N)
        m = Prophet()
        m.fit(train)
        self.assertNotIn('daily', m.seasonalities)
        m = Prophet(daily_seasonality=True)
        m.fit(train)
        self.assertIn('daily', m.seasonalities)
        m = Prophet(daily_seasonality=7, seasonality_prior_scale=3.)
        m.fit(DATA2)
        self.assertEqual(
            m.seasonalities['daily'],
            {
                'period': 1,
                'fourier_order': 7,
                'prior_scale': 3.,
                'mode': 'additive',
            },
        )
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
            'prior_scale': [4.],
        })
        m = Prophet(holidays=holidays)
        m.add_seasonality(name='monthly', period=30, fourier_order=5,
                          prior_scale=2.)
        self.assertEqual(
            m.seasonalities['monthly'],
            {
                'period': 30,
                'fourier_order': 5,
                'prior_scale': 2.,
                'mode': 'additive',
            },
        )
        with self.assertRaises(ValueError):
            m.add_seasonality(name='special_day', period=30, fourier_order=5)
        with self.assertRaises(ValueError):
            m.add_seasonality(name='trend', period=30, fourier_order=5)
        m.add_seasonality(name='weekly', period=30, fourier_order=5)
        # Test priors
        m = Prophet(
            holidays=holidays, yearly_seasonality=False,
            seasonality_mode='multiplicative',
        )
        m.add_seasonality(name='monthly', period=30, fourier_order=5,
                          prior_scale=2., mode='additive')
        m.fit(DATA.copy())
        self.assertEqual(m.seasonalities['monthly']['mode'], 'additive')
        self.assertEqual(m.seasonalities['weekly']['mode'], 'multiplicative')
        seasonal_features, prior_scales, component_cols, modes = (
            m.make_all_seasonality_features(m.history)
        )
        self.assertEqual(sum(component_cols['monthly']), 10)
        self.assertEqual(sum(component_cols['special_day']), 1)
        self.assertEqual(sum(component_cols['weekly']), 6)
        self.assertEqual(sum(component_cols['additive_terms']), 10)
        self.assertEqual(sum(component_cols['multiplicative_terms']), 7)
        if seasonal_features.columns[0] == 'monthly_delim_1':
            true = [2.] * 10 + [10.] * 6 + [4.]
            self.assertEqual(sum(component_cols['monthly'][:10]), 10)
            self.assertEqual(sum(component_cols['weekly'][10:16]), 6)
        else:
            true = [10.] * 6 + [2.] * 10 + [4.]
            self.assertEqual(sum(component_cols['weekly'][:6]), 6)
            self.assertEqual(sum(component_cols['monthly'][6:16]), 10)
        self.assertEqual(prior_scales, true)

    def test_added_regressors(self):
        m = Prophet()
        m.add_regressor('binary_feature', prior_scale=0.2)
        m.add_regressor('numeric_feature', prior_scale=0.5)
        m.add_regressor(
            'numeric_feature2', prior_scale=0.5, mode='multiplicative'
        )
        m.add_regressor('binary_feature2', standardize=True)
        df = DATA.copy()
        df['binary_feature'] = [0] * 255 + [1] * 255
        df['numeric_feature'] = range(510)
        df['numeric_feature2'] = range(510)
        with self.assertRaises(ValueError):
            # Require all regressors in df
            m.fit(df)
        df['binary_feature2'] = [1] * 100 + [0] * 410
        m.fit(df)
        # Check that standardizations are correctly set
        self.assertEqual(
            m.extra_regressors['binary_feature'],
            {
                'prior_scale': 0.2,
                'mu': 0,
                'std': 1,
                'standardize': 'auto',
                'mode': 'additive',
            },
        )
        self.assertEqual(
            m.extra_regressors['numeric_feature']['prior_scale'], 0.5)
        self.assertEqual(
            m.extra_regressors['numeric_feature']['mu'], 254.5)
        self.assertAlmostEqual(
            m.extra_regressors['numeric_feature']['std'], 147.368585, places=5)
        self.assertEqual(
            m.extra_regressors['numeric_feature2']['mode'], 'multiplicative')
        self.assertEqual(
            m.extra_regressors['binary_feature2']['prior_scale'], 10.)
        self.assertAlmostEqual(
            m.extra_regressors['binary_feature2']['mu'], 0.1960784, places=5)
        self.assertAlmostEqual(
            m.extra_regressors['binary_feature2']['std'], 0.3974183, places=5)
        # Check that standardization is done correctly
        df2 = m.setup_dataframe(df.copy())
        self.assertEqual(df2['binary_feature'][0], 0)
        self.assertAlmostEqual(df2['numeric_feature'][0], -1.726962, places=4)
        self.assertAlmostEqual(df2['binary_feature2'][0], 2.022859, places=4)
        # Check that feature matrix and prior scales are correctly constructed
        seasonal_features, prior_scales, component_cols, modes = (
            m.make_all_seasonality_features(df2)
        )
        self.assertEqual(seasonal_features.shape[1], 30)
        names = ['binary_feature', 'numeric_feature', 'binary_feature2']
        true_priors = [0.2, 0.5, 10.]
        for i, name in enumerate(names):
            self.assertIn(name, seasonal_features)
            self.assertEqual(sum(component_cols[name]), 1)
            self.assertEqual(
                sum(np.array(prior_scales) * component_cols[name]),
                true_priors[i],
            )
        # Check that forecast components are reasonable
        future = pd.DataFrame({
            'ds': ['2014-06-01'],
            'binary_feature': [0],
            'numeric_feature': [10],
            'numeric_feature2': [10],
        })
        with self.assertRaises(ValueError):
            m.predict(future)
        future['binary_feature2'] = 0
        fcst = m.predict(future)
        self.assertEqual(fcst.shape[1], 37)
        self.assertEqual(fcst['binary_feature'][0], 0)
        self.assertAlmostEqual(
            fcst['extra_regressors_additive'][0],
            fcst['numeric_feature'][0] + fcst['binary_feature2'][0],
        )
        self.assertAlmostEqual(
            fcst['extra_regressors_multiplicative'][0],
            fcst['numeric_feature2'][0],
        )
        self.assertAlmostEqual(
            fcst['additive_terms'][0],
            fcst['yearly'][0] + fcst['weekly'][0]
                + fcst['extra_regressors_additive'][0],
        )
        self.assertAlmostEqual(
            fcst['multiplicative_terms'][0],
            fcst['extra_regressors_multiplicative'][0],
        )
        self.assertAlmostEqual(
            fcst['yhat'][0],
            fcst['trend'][0] * (1 + fcst['multiplicative_terms'][0])
                + fcst['additive_terms'][0],
        )
        # Check works if constant extra regressor at 0
        df['constant_feature'] = 0
        m = Prophet()
        m.add_regressor('constant_feature')
        m.fit(df)
        self.assertEqual(m.extra_regressors['constant_feature']['std'], 1)

    def test_set_seasonality_mode(self):
        # Setting attribute
        m = Prophet()
        self.assertEqual(m.seasonality_mode, 'additive')
        m = Prophet(seasonality_mode='multiplicative')
        self.assertEqual(m.seasonality_mode, 'multiplicative')
        with self.assertRaises(ValueError):
            Prophet(seasonality_mode='batman')

    def test_seasonality_modes(self):
        # Model with holidays, seasonalities, and extra regressors
        holidays = pd.DataFrame({
            'ds': pd.to_datetime(['2016-12-25']),
            'holiday': ['xmas'],
            'lower_window': [-1],
            'upper_window': [0],
        })
        m = Prophet(seasonality_mode='multiplicative', holidays=holidays)
        m.add_seasonality('monthly', period=30, mode='additive', fourier_order=3)
        m.add_regressor('binary_feature', mode='additive')
        m.add_regressor('numeric_feature')
        # Construct seasonal features
        df = DATA.copy()
        df['binary_feature'] = [0] * 255 + [1] * 255
        df['numeric_feature'] = range(510)
        df = m.setup_dataframe(df, initialize_scales=True)
        m.history = df.copy()
        m.set_auto_seasonalities()
        seasonal_features, prior_scales, component_cols, modes = (
            m.make_all_seasonality_features(df))
        self.assertEqual(sum(component_cols['additive_terms']), 7)
        self.assertEqual(sum(component_cols['multiplicative_terms']), 29)
        self.assertEqual(
            set(modes['additive']),
            {'monthly', 'binary_feature', 'additive_terms',
             'extra_regressors_additive'},
        )
        self.assertEqual(
            set(modes['multiplicative']),
            {'weekly', 'yearly', 'xmas', 'numeric_feature',
             'multiplicative_terms', 'extra_regressors_multiplicative',
             'holidays',
            },
        )
