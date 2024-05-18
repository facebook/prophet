# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import itertools

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet, diagnostics


@pytest.fixture(scope="module")
def ts_short(daily_univariate_ts):
    return daily_univariate_ts.head(100)


class CustomParallelBackend:
    def map(self, func, *iterables):
        results = [func(*args) for args in zip(*iterables)]
        return results


PARALLEL_METHODS = [None, "processes", "threads", CustomParallelBackend()]
try:
    from dask.distributed import Client

    client = Client(processes=False)  # noqa
    PARALLEL_METHODS.append("dask")
except ImportError:
    pass


class TestCrossValidation:
    @pytest.mark.parametrize("parallel_method", PARALLEL_METHODS)
    def test_cross_validation(self, ts_short, parallel_method, backend):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)
        # Calculate the number of cutoff points(k)
        horizon = pd.Timedelta("4 days")
        period = pd.Timedelta("10 days")
        initial = pd.Timedelta("115 days")
        df_cv = diagnostics.cross_validation(
            m, horizon="4 days", period="10 days", initial="115 days", parallel=parallel_method
        )
        assert len(np.unique(df_cv["cutoff"])) == 3
        assert max(df_cv["ds"] - df_cv["cutoff"]) == horizon
        assert min(df_cv["cutoff"]) >= min(ts_short["ds"]) + initial
        dc = df_cv["cutoff"].diff()
        dc = dc[dc > pd.Timedelta(0)].min()
        assert dc >= period
        assert (df_cv["cutoff"] < df_cv["ds"]).all()
        # Each y in df_cv and ts_short with same ds should be equal
        df_merged = pd.merge(df_cv, ts_short, "left", on="ds")
        assert np.sum((df_merged["y_x"] - df_merged["y_y"]) ** 2) == pytest.approx(0.0)
        df_cv = diagnostics.cross_validation(
            m, horizon="4 days", period="10 days", initial="135 days"
        )
        assert len(np.unique(df_cv["cutoff"])) == 1
        with pytest.raises(ValueError):
            diagnostics.cross_validation(m, horizon="10 days", period="10 days", initial="140 days")

    def test_bad_parallel_methods(self, ts_short, backend):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)
        # invalid alias
        with pytest.raises(ValueError, match="'parallel' should be one"):
            diagnostics.cross_validation(m, horizon="4 days", parallel="bad")
        # no map method
        with pytest.raises(ValueError, match="'parallel' should be one"):
            diagnostics.cross_validation(m, horizon="4 days", parallel=object())

    def test_check_single_cutoff_forecast_func_calls(self, ts_short, monkeypatch, backend):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)

        def mock_predict(df, model, cutoff, horizon, predict_columns):
            nonlocal n_calls
            n_calls = n_calls + 1
            return pd.DataFrame(
                {
                    "ds": pd.date_range(start="2012-09-17", periods=3),
                    "yhat": np.arange(16, 19),
                    "yhat_lower": np.arange(15, 18),
                    "yhat_upper": np.arange(17, 20),
                    "y": np.arange(16.5, 19.5),
                    "cutoff": [datetime.date(2012, 9, 15)] * 3,
                }
            )

        monkeypatch.setattr(diagnostics, "single_cutoff_forecast", mock_predict)
        # cross validation  with 3 and 7 forecasts
        for args, forecasts in (
            (["4 days", "10 days", "115 days"], 3),
            (["4 days", "4 days", "115 days"], 7),
        ):
            n_calls = 0
            _ = diagnostics.cross_validation(m, *args)
            # check single forecast function called expected number of times
            assert n_calls == forecasts
    
    @pytest.mark.parametrize("extra_output_columns", ["trend", ["trend"]])
    def test_check_extra_output_columns_cross_validation(self, ts_short, backend, extra_output_columns):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)
        df_cv = diagnostics.cross_validation(
            m,
            horizon="1 days",
            period="1 days",
            initial="140 days",
            extra_output_columns=extra_output_columns
        )
        assert "trend" in df_cv.columns

    @pytest.mark.parametrize("growth", ["logistic", "flat"])
    def test_cross_validation_logistic_or_flat_growth(self, growth, ts_short, backend):
        df = ts_short.copy()
        if growth == "logistic":
            df["cap"] = 40
        m = Prophet(growth=growth, stan_backend=backend).fit(df)
        df_cv = diagnostics.cross_validation(
            m, horizon="1 days", period="1 days", initial="140 days"
        )
        assert len(np.unique(df_cv["cutoff"])) == 2
        assert (df_cv["cutoff"] < df_cv["ds"]).all()
        df_merged = pd.merge(df_cv, ts_short, "left", on="ds")
        assert np.sum((df_merged["y_x"] - df_merged["y_y"]) ** 2) == pytest.approx(0.0)

    def test_cross_validation_extra_regressors(self, ts_short, backend):
        df = ts_short.copy()
        df["extra"] = range(df.shape[0])
        df["is_conditional_week"] = np.arange(df.shape[0]) // 7 % 2
        m = Prophet(stan_backend=backend)
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m.add_seasonality(
            name="conditional_weekly",
            period=7,
            fourier_order=3,
            prior_scale=2.0,
            condition_name="is_conditional_week",
        )
        m.add_regressor("extra")
        m.fit(df)
        df_cv = diagnostics.cross_validation(
            m, horizon="4 days", period="4 days", initial="135 days"
        )
        assert len(np.unique(df_cv["cutoff"])) == 2
        period = pd.Timedelta("4 days")
        dc = df_cv["cutoff"].diff()
        dc = dc[dc > pd.Timedelta(0)].min()
        assert dc >= period
        assert (df_cv["cutoff"] < df_cv["ds"]).all()
        df_merged = pd.merge(df_cv, ts_short, "left", on="ds")
        assert np.sum((df_merged["y_x"] - df_merged["y_y"]) ** 2) == pytest.approx(0.0)

    def test_cross_validation_default_value_check(self, ts_short, backend):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)
        # Default value of initial should be equal to 3 * horizon
        df_cv1 = diagnostics.cross_validation(m, horizon="32 days", period="10 days")
        df_cv2 = diagnostics.cross_validation(
            m, horizon="32 days", period="10 days", initial="96 days"
        )
        assert ((df_cv1["y"] - df_cv2["y"]) ** 2).sum() == pytest.approx(0.0)
        assert ((df_cv1["yhat"] - df_cv2["yhat"]) ** 2).sum() == pytest.approx(0.0)

    def test_cross_validation_custom_cutoffs(self, ts_short, backend):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)
        # When specify a list of cutoffs
        #  the cutoff dates in df_cv are those specified
        df_cv1 = diagnostics.cross_validation(
            m,
            horizon="32 days",
            period="10 days",
            cutoffs=[pd.Timestamp("2012-07-31"), pd.Timestamp("2012-08-31")],
        )
        assert len(df_cv1["cutoff"].unique()) == 2

    def test_cross_validation_uncertainty_disabled(self, ts_short, backend):
        df = ts_short.copy()
        for uncertainty in [0, False]:
            m = Prophet(uncertainty_samples=uncertainty, stan_backend=backend)
            m.fit(df, algorithm="Newton")
            df_cv = diagnostics.cross_validation(
                m, horizon="4 days", period="4 days", initial="115 days"
            )
            expected_cols = ["ds", "yhat", "y", "cutoff"]
            assert all(col in expected_cols for col in df_cv.columns.tolist())
            df_p = diagnostics.performance_metrics(df_cv)
            assert "coverage" not in df_p.columns


class TestPerformanceMetrics:
    def test_performance_metrics(self, ts_short, backend):
        m = Prophet(stan_backend=backend)
        m.fit(ts_short)
        df_cv = diagnostics.cross_validation(
            m, horizon="4 days", period="10 days", initial="90 days"
        )
        # Aggregation level none
        df_none = diagnostics.performance_metrics(df_cv, rolling_window=-1)
        assert set(df_none.columns) == {
            "horizon",
            "coverage",
            "mae",
            "mape",
            "mdape",
            "mse",
            "rmse",
            "smape",
        }
        assert df_none.shape[0] == 16
        # Aggregation level 0
        df_0 = diagnostics.performance_metrics(df_cv, rolling_window=0)
        assert len(df_0) == 4
        assert len(df_0["horizon"].unique()) == 4
        # Aggregation level 0.2
        df_horizon = diagnostics.performance_metrics(df_cv, rolling_window=0.2)
        assert len(df_horizon) == 4
        assert len(df_horizon["horizon"].unique()) == 4
        # Aggregation level all
        df_all = diagnostics.performance_metrics(df_cv, rolling_window=1)
        assert df_all.shape[0] == 1
        for metric in ["mse", "mape", "mae", "coverage"]:
            assert df_all[metric].values[0] == pytest.approx(df_none[metric].mean())
        assert df_all["mdape"].values[0] == pytest.approx(df_none["mdape"].median())
        # Custom list of metrics
        df_horizon = diagnostics.performance_metrics(
            df_cv,
            metrics=["coverage", "mse"],
        )
        assert set(df_horizon.columns) == {"coverage", "mse", "horizon"}
        # Skip MAPE
        df_cv.loc[0, "y"] = 0.0
        df_horizon = diagnostics.performance_metrics(
            df_cv,
            metrics=["coverage", "mape"],
        )
        assert set(df_horizon.columns) == {"coverage", "horizon"}
        # Handle zero y and yhat
        df_cv["y"] = 0.0
        df_cv["yhat"] = 0.0
        df_horizon = diagnostics.performance_metrics(
            df_cv,
        )
        assert set(df_horizon.columns) == {"coverage", "horizon", "mae", "mdape", "mse", "rmse", "smape"}
        df_horizon = diagnostics.performance_metrics(
            df_cv,
            metrics=["mape"],
        )
        assert df_horizon is None
        # List of metrics containing non-valid metrics
        with pytest.raises(ValueError):
            diagnostics.performance_metrics(
                df_cv,
                metrics=["mse", "error_metric"],
            )

    def test_rolling_mean(self):
        x = np.arange(10)
        h = np.arange(10)
        df = diagnostics.rolling_mean_by_h(x=x, h=h, w=1, name="x")
        assert np.array_equal(x, df["x"].values)
        assert np.array_equal(h, df["horizon"].values)

        df = diagnostics.rolling_mean_by_h(x, h, w=4, name="x")
        assert np.allclose(x[3:] - 1.5, df["x"].values)
        assert np.array_equal(np.arange(3, 10), df["horizon"].values)

        h = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0])
        x_true = np.array([1.0, 5.0, 22.0 / 3])
        h_true = np.array([3.0, 4.0, 7.0])
        df = diagnostics.rolling_mean_by_h(x, h, w=3, name="x")
        assert np.allclose(x_true, df["x"].values)
        assert np.array_equal(h_true, df["horizon"].values)

        df = diagnostics.rolling_mean_by_h(x, h, w=10, name="x")
        assert np.allclose(np.array([7.0]), df["horizon"].values)
        assert np.allclose(np.array([4.5]), df["x"].values)

    def test_rolling_median(self):
        x = np.arange(10)
        h = np.arange(10)
        df = diagnostics.rolling_median_by_h(x=x, h=h, w=1, name="x")
        assert np.array_equal(x, df["x"].values)
        assert np.array_equal(h, df["horizon"].values)

        df = diagnostics.rolling_median_by_h(x, h, w=4, name="x")
        x_true = x[3:] - 1.5
        assert np.allclose(x_true, df["x"].values)
        assert np.array_equal(np.arange(3, 10), df["horizon"].values)

        h = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 7.0])
        x_true = np.array([1.0, 5.0, 8.0])
        h_true = np.array([3.0, 4.0, 7.0])
        df = diagnostics.rolling_median_by_h(x, h, w=3, name="x")
        assert np.allclose(x_true, df["x"].values)
        assert np.array_equal(h_true, df["horizon"].values)

        df = diagnostics.rolling_median_by_h(x, h, w=10, name="x")
        assert np.allclose(np.array([7.0]), df["horizon"].values)
        assert np.allclose(np.array([4.5]), df["x"].values)


class TestProphetCopy:
    @pytest.fixture(scope="class")
    def data(self, daily_univariate_ts):
        df = daily_univariate_ts.copy()
        df["cap"] = 200.0
        df["binary_feature"] = [0] * 255 + [1] * 255
        return df

    def test_prophet_copy(self, data, backend):
        # These values are created except for its default values
        holiday = pd.DataFrame({"ds": pd.to_datetime(["2016-12-25"]), "holiday": ["x"]})
        products = itertools.product(
            ["linear", "logistic"],  # growth
            [None, pd.to_datetime(["2016-12-25"])],  # changepoints
            [3],  # n_changepoints
            [0.9],  # changepoint_range
            [True, False],  # yearly_seasonality
            [True, False],  # weekly_seasonality
            [True, False],  # daily_seasonality
            [None, holiday],  # holidays
            ["additive", "multiplicative"],  # seasonality_mode
            [1.1],  # seasonality_prior_scale
            [1.1],  # holidays_prior_scale
            [0.1],  # changepoint_prior_scale
            [100],  # mcmc_samples
            [0.9],  # interval_width
            [200],  # uncertainty_samples
        )
        # Values should be copied correctly
        for product in products:
            m1 = Prophet(*product, stan_backend=backend)
            m1.country_holidays = "US"
            m1.history = m1.setup_dataframe(data.copy(), initialize_scales=True)
            m1.set_auto_seasonalities()
            m2 = diagnostics.prophet_copy(m1)
            assert m1.growth == m2.growth
            assert m1.n_changepoints == m2.n_changepoints
            assert m1.changepoint_range == m2.changepoint_range
            if m1.changepoints is None:
                assert m1.changepoints == m2.changepoints
            else:
                assert m1.changepoints.equals(m2.changepoints)
            assert False == m2.yearly_seasonality
            assert False == m2.weekly_seasonality
            assert False == m2.daily_seasonality
            assert m1.yearly_seasonality == ("yearly" in m2.seasonalities)
            assert m1.weekly_seasonality == ("weekly" in m2.seasonalities)
            assert m1.daily_seasonality == ("daily" in m2.seasonalities)
            if m1.holidays is None:
                assert m1.holidays == m2.holidays
            else:
                assert (m1.holidays == m2.holidays).values.all()
            assert m1.country_holidays == m2.country_holidays
            assert m1.holidays_mode == m2.holidays_mode
            assert m1.seasonality_mode == m2.seasonality_mode
            assert m1.seasonality_prior_scale == m2.seasonality_prior_scale
            assert m1.changepoint_prior_scale == m2.changepoint_prior_scale
            assert m1.holidays_prior_scale == m2.holidays_prior_scale
            assert m1.mcmc_samples == m2.mcmc_samples
            assert m1.interval_width == m2.interval_width
            assert m1.uncertainty_samples == m2.uncertainty_samples

    def test_prophet_copy_custom(self, data, backend):
        changepoints = pd.date_range("2012-06-15", "2012-09-15")
        cutoff = pd.Timestamp("2012-07-25")
        m1 = Prophet(changepoints=changepoints, stan_backend=backend)
        m1.add_seasonality("custom", 10, 5)
        m1.add_regressor("binary_feature")
        m1.fit(data)
        m2 = diagnostics.prophet_copy(m1, cutoff=cutoff)
        changepoints = changepoints[changepoints < cutoff]
        assert (changepoints == m2.changepoints).all()
        assert "custom" in m2.seasonalities
        assert "binary_feature" in m2.extra_regressors
