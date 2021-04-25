# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import datetime

from prophet import Prophet
from prophet import diagnostics

DATA_all = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'data.csv'), parse_dates=['ds']
)
DATA = DATA_all.head(100)


class CustomParallelBackend:
    def map(self, func, *iterables):
        results = [func(*args) for args in zip(*iterables)]
        return results


class TestDiagnostics(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDiagnostics, self).__init__(*args, **kwargs)
        # Use first 100 record in data.csv
        self.__df = DATA

    def test_cross_validation(self):
        m = Prophet()
        m.fit(self.__df)
        # Calculate the number of cutoff points(k)
        horizon = pd.Timedelta('4 days')
        period = pd.Timedelta('10 days')
        initial = pd.Timedelta('115 days')
        methods = [None, 'processes', 'threads', CustomParallelBackend()]

        try:
            from dask.distributed import Client
            client = Client(processes=False)  # noqa
            methods.append("dask")
        except ImportError:
            pass

        for parallel in methods:
            df_cv = diagnostics.cross_validation(
                m, horizon='4 days', period='10 days', initial='115 days',
                parallel=parallel)
            self.assertEqual(len(np.unique(df_cv['cutoff'])), 3)
            self.assertEqual(max(df_cv['ds'] - df_cv['cutoff']), horizon)
            self.assertTrue(min(df_cv['cutoff']) >= min(self.__df['ds']) + initial)
            dc = df_cv['cutoff'].diff()
            dc = dc[dc > pd.Timedelta(0)].min()
            self.assertTrue(dc >= period)
            self.assertTrue((df_cv['cutoff'] < df_cv['ds']).all())
            # Each y in df_cv and self.__df with same ds should be equal
            df_merged = pd.merge(df_cv, self.__df, 'left', on='ds')
            self.assertAlmostEqual(
                np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 0.0)
            df_cv = diagnostics.cross_validation(
                m, horizon='4 days', period='10 days', initial='135 days')
            self.assertEqual(len(np.unique(df_cv['cutoff'])), 1)
            with self.assertRaises(ValueError):
                diagnostics.cross_validation(
                    m, horizon='10 days', period='10 days', initial='140 days')

        # invalid alias
        with self.assertRaisesRegex(ValueError, "'parallel' should be one"):
            diagnostics.cross_validation(m, horizon="4 days", parallel="bad")

        # no map method
        with self.assertRaisesRegex(ValueError, "'parallel' should be one"):
            diagnostics.cross_validation(m, horizon="4 days", parallel=object())


    def test_check_single_cutoff_forecast_func_calls(self):
        m = Prophet()
        m.fit(self.__df)
        mock_predict = pd.DataFrame({'ds':pd.date_range(start='2012-09-17', periods=3),
                                     'yhat':np.arange(16, 19),
                                     'yhat_lower':np.arange(15, 18),
                                     'yhat_upper': np.arange(17, 20),
                                      'y': np.arange(16.5, 19.5),
                                     'cutoff': [datetime.date(2012, 9, 15)]*3})

        # cross validation  with 3 and 7 forecasts
        for args, forecasts in ((['4 days', '10 days', '115 days'], 3),
                            (['4 days', '4 days', '115 days'], 7)):
            with patch('prophet.diagnostics.single_cutoff_forecast') as mock_func:
                mock_func.return_value = mock_predict
                df_cv = diagnostics.cross_validation(m, *args)
                # check single forecast function called expected number of times
                self.assertEqual(diagnostics.single_cutoff_forecast.call_count,
                                 forecasts)

    def test_cross_validation_logistic_or_flat_growth(self):
        params = (x for x in ['logistic', 'flat'])
        for growth in params:
            with self.subTest(i=growth):
                df = self.__df.copy()
                if growth == "logistic":
                    df['cap'] = 40
                m = Prophet(growth=growth).fit(df)
                df_cv = diagnostics.cross_validation(
                    m, horizon='1 days', period='1 days', initial='140 days')
                self.assertEqual(len(np.unique(df_cv['cutoff'])), 2)
                self.assertTrue((df_cv['cutoff'] < df_cv['ds']).all())
                df_merged = pd.merge(df_cv, self.__df, 'left', on='ds')
                self.assertAlmostEqual(
                    np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 0.0)

    def test_cross_validation_extra_regressors(self):
        df = self.__df.copy()
        df['extra'] = range(df.shape[0])
        df['is_conditional_week'] = np.arange(df.shape[0]) // 7 % 2
        m = Prophet()
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        m.add_seasonality(name='conditional_weekly', period=7, fourier_order=3,
                          prior_scale=2., condition_name='is_conditional_week')
        m.add_regressor('extra')
        m.fit(df)
        df_cv = diagnostics.cross_validation(
            m, horizon='4 days', period='4 days', initial='135 days')
        self.assertEqual(len(np.unique(df_cv['cutoff'])), 2)
        period = pd.Timedelta('4 days')
        dc = df_cv['cutoff'].diff()
        dc = dc[dc > pd.Timedelta(0)].min()
        self.assertTrue(dc >= period)
        self.assertTrue((df_cv['cutoff'] < df_cv['ds']).all())
        df_merged = pd.merge(df_cv, self.__df, 'left', on='ds')
        self.assertAlmostEqual(
            np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 0.0)

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

    def test_cross_validation_custom_cutoffs(self):
        m = Prophet()
        m.fit(self.__df)
        # When specify a list of cutoffs
        #  the cutoff dates in df_cv are those specified
        df_cv1 = diagnostics.cross_validation(
            m,
            horizon='32 days',
            period='10 days',
            cutoffs=[pd.Timestamp('2012-07-31'), pd.Timestamp('2012-08-31')])
        self.assertEqual(len(df_cv1['cutoff'].unique()), 2)
      
    def test_cross_validation_uncertainty_disabled(self):
        df = self.__df.copy()
        for uncertainty in [0, False]:
            m = Prophet(uncertainty_samples=uncertainty)
            m.fit(df, algorithm='Newton')
            df_cv = diagnostics.cross_validation(
                m, horizon='4 days', period='4 days', initial='115 days')
            expected_cols = ['ds', 'yhat', 'y', 'cutoff']
            self.assertTrue(all(col in expected_cols for col in df_cv.columns.tolist()))
            df_p = diagnostics.performance_metrics(df_cv)
            self.assertTrue('coverage' not in df_p.columns)

    def test_performance_metrics(self):
        m = Prophet()
        m.fit(self.__df)
        df_cv = diagnostics.cross_validation(
            m, horizon='4 days', period='10 days', initial='90 days')
        # Aggregation level none
        df_none = diagnostics.performance_metrics(df_cv, rolling_window=-1)
        self.assertEqual(
            set(df_none.columns),
            {'horizon', 'coverage', 'mae', 'mape', 'mdape', 'mse', 'rmse', 'smape'},
        )
        self.assertEqual(df_none.shape[0], 16)
        # Aggregation level 0
        df_0 = diagnostics.performance_metrics(df_cv, rolling_window=0)
        self.assertEqual(len(df_0), 4)
        self.assertEqual(len(df_0['horizon'].unique()), 4)
        # Aggregation level 0.2
        df_horizon = diagnostics.performance_metrics(df_cv, rolling_window=0.2)
        self.assertEqual(len(df_horizon), 4)
        self.assertEqual(len(df_horizon['horizon'].unique()), 4)
        # Aggregation level all
        df_all = diagnostics.performance_metrics(df_cv, rolling_window=1)
        self.assertEqual(df_all.shape[0], 1)
        for metric in ['mse', 'mape', 'mae', 'coverage']:
            self.assertAlmostEqual(df_all[metric].values[0], df_none[metric].mean())
        self.assertAlmostEqual(df_all['mdape'].values[0], df_none['mdape'].median())
        # Custom list of metrics
        df_horizon = diagnostics.performance_metrics(
            df_cv, metrics=['coverage', 'mse'],
        )
        self.assertEqual(
            set(df_horizon.columns),
            {'coverage', 'mse', 'horizon'},
        )
        # Skip MAPE
        df_cv.loc[0, 'y'] = 0.
        df_horizon = diagnostics.performance_metrics(
            df_cv, metrics=['coverage', 'mape'],
        )
        self.assertEqual(
            set(df_horizon.columns),
            {'coverage', 'horizon'},
        )
        df_horizon = diagnostics.performance_metrics(
            df_cv, metrics=['mape'],
        )
        self.assertIsNone(df_horizon)
        # List of metrics containing non-valid metrics
        with self.assertRaises(ValueError):
            diagnostics.performance_metrics(
                df_cv, metrics=['mse', 'error_metric'],
            )

    def test_rolling_mean(self):
        x = np.arange(10)
        h = np.arange(10)
        df = diagnostics.rolling_mean_by_h(x=x, h=h, w=1, name='x')
        self.assertTrue(np.array_equal(x, df['x'].values))
        self.assertTrue(np.array_equal(h, df['horizon'].values))

        df = diagnostics.rolling_mean_by_h(x, h, w=4, name='x')
        self.assertTrue(np.allclose(x[3:] - 1.5, df['x'].values))
        self.assertTrue(np.array_equal(np.arange(3, 10), df['horizon'].values))

        h = np.array([1., 2., 3., 4., 4., 4., 4., 4., 7., 7.])
        x_true = np.array([1.0, 5.0 , 22. / 3])
        h_true = np.array([3., 4., 7.])
        df = diagnostics.rolling_mean_by_h(x, h, w=3, name='x')
        self.assertTrue(np.allclose(x_true, df['x'].values))
        self.assertTrue(np.array_equal(h_true, df['horizon'].values))

        df = diagnostics.rolling_mean_by_h(x, h, w=10, name='x')
        self.assertTrue(np.allclose(np.array([7.]), df['horizon'].values))
        self.assertTrue(np.allclose(np.array([4.5]), df['x'].values))

    def test_rolling_median(self):
        x = np.arange(10)
        h = np.arange(10)
        df = diagnostics.rolling_median_by_h(x=x, h=h, w=1, name='x')
        self.assertTrue(np.array_equal(x, df['x'].values))
        self.assertTrue(np.array_equal(h, df['horizon'].values))

        df = diagnostics.rolling_median_by_h(x, h, w=4, name='x')
        x_true = x[3:] - 1.5
        self.assertTrue(np.allclose(x_true, df['x'].values))
        self.assertTrue(np.array_equal(np.arange(3, 10), df['horizon'].values))

        h = np.array([1., 2., 3., 4., 4., 4., 4., 4., 7., 7.])
        x_true = np.array([1.0, 5.0, 8.0])
        h_true = np.array([3., 4., 7.])
        df = diagnostics.rolling_median_by_h(x, h, w=3, name='x')
        self.assertTrue(np.allclose(x_true, df['x'].values))
        self.assertTrue(np.array_equal(h_true, df['horizon'].values))

        df = diagnostics.rolling_median_by_h(x, h, w=10, name='x')
        self.assertTrue(np.allclose(np.array([7.]), df['horizon'].values))
        self.assertTrue(np.allclose(np.array([4.5]), df['x'].values))

    def test_copy(self):
        df = DATA_all.copy()
        df['cap'] = 200.
        df['binary_feature'] = [0] * 255 + [1] * 255
        # These values are created except for its default values
        holiday = pd.DataFrame(
            {'ds': pd.to_datetime(['2016-12-25']), 'holiday': ['x']})
        products = itertools.product(
            ['linear', 'logistic'],  # growth
            [None, pd.to_datetime(['2016-12-25'])],  # changepoints
            [3],  # n_changepoints
            [0.9],  # changepoint_range
            [True, False],  # yearly_seasonality
            [True, False],  # weekly_seasonality
            [True, False],  # daily_seasonality
            [None, holiday],  # holidays
            ['additive', 'multiplicative'],  # seasonality_mode
            [1.1],  # seasonality_prior_scale
            [1.1],  # holidays_prior_scale
            [0.1],  # changepoint_prior_scale
            [100],  # mcmc_samples
            [0.9],  # interval_width
            [200]  # uncertainty_samples
        )
        # Values should be copied correctly
        for product in products:
            m1 = Prophet(*product)
            m1.country_holidays = 'US'
            m1.history = m1.setup_dataframe(
                df.copy(), initialize_scales=True)
            m1.set_auto_seasonalities()
            m2 = diagnostics.prophet_copy(m1)
            self.assertEqual(m1.growth, m2.growth)
            self.assertEqual(m1.n_changepoints, m2.n_changepoints)
            self.assertEqual(m1.changepoint_range, m2.changepoint_range)
            if m1.changepoints is None:
                self.assertEqual(m1.changepoints, m2.changepoints)
            else:
                self.assertTrue(m1.changepoints.equals(m2.changepoints))
            self.assertEqual(False, m2.yearly_seasonality)
            self.assertEqual(False, m2.weekly_seasonality)
            self.assertEqual(False, m2.daily_seasonality)
            self.assertEqual(
                m1.yearly_seasonality, 'yearly' in m2.seasonalities)
            self.assertEqual(
                m1.weekly_seasonality, 'weekly' in m2.seasonalities)
            self.assertEqual(
                m1.daily_seasonality, 'daily' in m2.seasonalities)
            if m1.holidays is None:
                self.assertEqual(m1.holidays, m2.holidays)
            else:
                self.assertTrue((m1.holidays == m2.holidays).values.all())
            self.assertEqual(m1.country_holidays, m2.country_holidays)
            self.assertEqual(m1.seasonality_mode, m2.seasonality_mode)
            self.assertEqual(m1.seasonality_prior_scale,
                             m2.seasonality_prior_scale)
            self.assertEqual(m1.changepoint_prior_scale,
                             m2.changepoint_prior_scale)
            self.assertEqual(m1.holidays_prior_scale, m2.holidays_prior_scale)
            self.assertEqual(m1.mcmc_samples, m2.mcmc_samples)
            self.assertEqual(m1.interval_width, m2.interval_width)
            self.assertEqual(m1.uncertainty_samples, m2.uncertainty_samples)

        # Check for cutoff and custom seasonality and extra regressors
        changepoints = pd.date_range('2012-06-15', '2012-09-15')
        cutoff = pd.Timestamp('2012-07-25')
        m1 = Prophet(changepoints=changepoints)
        m1.add_seasonality('custom', 10, 5)
        m1.add_regressor('binary_feature')
        m1.fit(df)
        m2 = diagnostics.prophet_copy(m1, cutoff=cutoff)
        changepoints = changepoints[changepoints < cutoff]
        self.assertTrue((changepoints == m2.changepoints).all())
        self.assertTrue('custom' in m2.seasonalities)
        self.assertTrue('binary_feature' in m2.extra_regressors)

