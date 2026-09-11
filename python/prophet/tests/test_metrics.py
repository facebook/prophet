# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest

from prophet import metrics


class TestRollingMeanByH:
    def test_basic(self):
        x = np.arange(10)
        h = np.arange(10)
        df = metrics.rolling_mean_by_h(x=x, h=h, w=1, name="x")
        assert np.array_equal(x, df["x"].values)
        assert np.array_equal(h, df["horizon"].values)

    def test_window_4(self):
        x = np.arange(10)
        h = np.arange(10)
        df = metrics.rolling_mean_by_h(x, h, w=4, name="x")
        assert np.allclose(x[3:] - 1.5, df["x"].values)
        assert np.array_equal(np.arange(3, 10), df["horizon"].values)

    def test_grouped_horizons(self):
        x = np.arange(10)
        h = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0])
        x_true = np.array([1.0, 5.0, 22.0 / 3])
        h_true = np.array([3.0, 4.0, 7.0])
        df = metrics.rolling_mean_by_h(x, h, w=3, name="x")
        assert np.allclose(x_true, df["x"].values)
        assert np.array_equal(h_true, df["horizon"].values)

    def test_large_window(self):
        x = np.arange(10)
        h = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0])
        df = metrics.rolling_mean_by_h(x, h, w=10, name="x")
        assert np.allclose(np.array([7.0]), df["horizon"].values)
        assert np.allclose(np.array([4.5]), df["x"].values)


class TestRollingMedianByH:
    def test_basic(self):
        x = np.arange(10)
        h = np.arange(10)
        df = metrics.rolling_median_by_h(x=x, h=h, w=1, name="x")
        assert np.array_equal(x, df["x"].values)
        assert np.array_equal(h, df["horizon"].values)

    def test_window_4(self):
        x = np.arange(10)
        h = np.arange(10)
        df = metrics.rolling_median_by_h(x, h, w=4, name="x")
        x_true = x[3:] - 1.5
        assert np.allclose(x_true, df["x"].values)
        assert np.array_equal(np.arange(3, 10), df["horizon"].values)

    def test_grouped_horizons(self):
        x = np.arange(10)
        h = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0])
        x_true = np.array([1.0, 5.0, 8.0])
        h_true = np.array([3.0, 4.0, 7.0])
        df = metrics.rolling_median_by_h(x, h, w=3, name="x")
        assert np.allclose(x_true, df["x"].values)
        assert np.array_equal(h_true, df["horizon"].values)

    def test_large_window(self):
        x = np.arange(10)
        h = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0])
        df = metrics.rolling_median_by_h(x, h, w=10, name="x")
        assert np.allclose(np.array([7.0]), df["horizon"].values)
        assert np.allclose(np.array([4.5]), df["x"].values)


class TestMetricsRegistered:
    def test_all_registered(self):
        expected = {'mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'}
        assert expected == set(metrics.PERFORMANCE_METRICS.keys())

    def test_register_custom(self):
        @metrics.register_performance_metric
        def custom_metric(df, w):
            return pd.DataFrame({'horizon': df['horizon'], 'custom': 0.0})

        assert 'custom_metric' in metrics.PERFORMANCE_METRICS
        assert metrics.PERFORMANCE_METRICS['custom_metric'] is custom_metric
        del metrics.PERFORMANCE_METRICS['custom_metric']


class TestPerformanceMetrics:
    def _make_cv_df(self):
        np.random.seed(42)
        ds = pd.date_range("2020-01-01", periods=20, freq="D")
        y = np.random.randn(20).cumsum() + 10
        return pd.DataFrame({
            'ds': ds,
            'y': y,
            'yhat': y + np.random.randn(20) * 0.5,
            'yhat_lower': y - 1,
            'yhat_upper': y + 1,
            'cutoff': [ds[0]] * 20,
        })

    def test_basic(self):
        df = self._make_cv_df()
        res = metrics.performance_metrics(df, rolling_window=0)
        assert res is not None
        assert 'horizon' in res.columns

    def test_all_metrics(self):
        df = self._make_cv_df()
        res = metrics.performance_metrics(df, rolling_window=-1)
        assert res is not None
        assert set(res.columns) == {
            'horizon', 'coverage', 'mae', 'mape', 'mdape', 'mse', 'rmse', 'smape'
        }

    def test_no_coverage_without_uncertainty(self):
        df = self._make_cv_df()
        df = df.drop(columns=['yhat_lower', 'yhat_upper'])
        res = metrics.performance_metrics(df)
        assert res is not None
        assert 'coverage' not in res.columns

    def test_skip_mape_near_zero(self):
        df = self._make_cv_df()
        df['y'] = 0.0
        df['yhat'] = 0.0
        res = metrics.performance_metrics(df, metrics=['mape'])
        assert res is None

    def test_invalid_metric(self):
        df = self._make_cv_df()
        with pytest.raises(ValueError):
            metrics.performance_metrics(df, metrics=['mse', 'invalid'])

    def test_duplicate_metrics(self):
        df = self._make_cv_df()
        with pytest.raises(ValueError):
            metrics.performance_metrics(df, metrics=['mse', 'mse'])

    def test_monthly_horizon(self):
        ds = pd.date_range("2020-01-01", periods=12, freq="MS")
        y = np.arange(12, dtype=float)
        cutoff = pd.Timestamp("2020-01-01")
        df = pd.DataFrame({
            'ds': ds,
            'y': y,
            'yhat': y + 0.5,
            'yhat_lower': y - 1,
            'yhat_upper': y + 1,
            'cutoff': [cutoff] * len(ds),
        })
        res = metrics.performance_metrics(df, monthly=True, rolling_window=0)
        assert res is not None
        assert res['horizon'].dtype in (np.int64, int)


class TestBackwardCompatibility:
    def test_import_from_diagnostics(self):
        from prophet.diagnostics import (
            PERFORMANCE_METRICS,
            coverage,
            mae,
            mape,
            mdape,
            mse,
            performance_metrics,
            register_performance_metric,
            rmse,
            rolling_mean_by_h,
            rolling_median_by_h,
            smape,
        )
        assert callable(mse)
        assert callable(performance_metrics)
        assert isinstance(PERFORMANCE_METRICS, dict)

    def test_import_from_metrics(self):
        assert callable(metrics.mse)
        assert callable(metrics.performance_metrics)
        assert isinstance(metrics.PERFORMANCE_METRICS, dict)
