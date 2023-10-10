# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet
from prophet.utilities import warm_start_params


def train_test_split(ts_data: pd.DataFrame, n_test_rows: int) -> pd.DataFrame:
    train = ts_data.head(ts_data.shape[0] - n_test_rows)
    test = ts_data.tail(n_test_rows)
    return train.reset_index(), test.reset_index()


def rmse(predictions, targets) -> float:
    return np.sqrt(np.mean((predictions - targets) ** 2))


class TestProphetFitPredictDefault:
    @pytest.mark.parametrize(
        "scaling,expected",
        [("absmax", 10.64), ("minmax", 11.13)],
        ids=["absmax", "minmax"]
    )
    def test_fit_predict(self, daily_univariate_ts, backend, scaling, expected):
        test_days = 30
        train, test = train_test_split(daily_univariate_ts, test_days)
        forecaster = Prophet(stan_backend=backend, scaling=scaling)
        forecaster.fit(train, seed=1237861298)
        np.random.seed(876543987)
        future = forecaster.make_future_dataframe(test_days, include_history=False)
        future = forecaster.predict(future)
        res = rmse(future["yhat"], test["y"])
        # Higher threshold due to cmdstan 2.33.1 producing numerical differences for macOS Intel (ARM is fine).
        assert res == pytest.approx(expected, 0.1), "backend: {}".format(forecaster.stan_backend)

    @pytest.mark.parametrize(
        "scaling,expected",
        [("absmax", 23.44), ("minmax", 11.29)],
        ids=["absmax", "minmax"]
    )
    def test_fit_predict_newton(self, daily_univariate_ts, backend, scaling, expected):
        test_days = 30
        train, test = train_test_split(daily_univariate_ts, test_days)
        forecaster = Prophet(stan_backend=backend, scaling=scaling)
        forecaster.fit(train, algorithm="Newton", seed=1237861298)
        np.random.seed(876543987)
        future = forecaster.make_future_dataframe(test_days, include_history=False)
        future = forecaster.predict(future)
        res = rmse(future["yhat"], test["y"])
        assert res == pytest.approx(expected, 0.01), "backend: {}".format(forecaster.stan_backend)

    @pytest.mark.parametrize(
        "scaling,expected",
        [("absmax", 127.01), ("minmax", 93.45)],
        ids=["absmax", "minmax"]
    )
    def test_fit_predict_large_numbers(self, large_numbers_ts, backend, scaling, expected):
        test_days = 30
        train, test = train_test_split(large_numbers_ts, test_days)
        forecaster = Prophet(stan_backend=backend, scaling=scaling)
        forecaster.fit(train, seed=1237861298)
        np.random.seed(876543987)
        future = forecaster.make_future_dataframe(test_days, include_history=False)
        future = forecaster.predict(future)
        res = rmse(future["yhat"], test["y"])
        assert res == pytest.approx(expected, 0.01), "backend: {}".format(forecaster.stan_backend)

    @pytest.mark.slow
    def test_fit_predict_sampling(self, daily_univariate_ts, backend):
        test_days = 30
        train, test = train_test_split(daily_univariate_ts, test_days)
        forecaster = Prophet(mcmc_samples=500, stan_backend=backend)
        # chains adjusted from 4 to 7 to satisfy test for cmdstanpy
        forecaster.fit(train, seed=1237861298, chains=7, show_progress=False)
        np.random.seed(876543987)
        future = forecaster.make_future_dataframe(test_days, include_history=False)
        future = forecaster.predict(future)
        # this gives ~ 215.77
        res = rmse(future["yhat"], test["y"])
        assert 236 < res < 193, "backend: {}".format(forecaster.stan_backend)

    def test_fit_predict_no_seasons(self, daily_univariate_ts, backend):
        test_days = 30
        train, _ = train_test_split(daily_univariate_ts, test_days)
        forecaster = Prophet(
            weekly_seasonality=False, yearly_seasonality=False, stan_backend=backend
        )
        forecaster.fit(train)
        future = forecaster.make_future_dataframe(test_days, include_history=False)
        result = forecaster.predict(future)
        assert (future.ds == result.ds).all()

    def test_fit_predict_no_changepoints(self, daily_univariate_ts, backend):
        test_days = daily_univariate_ts.shape[0] // 2
        train, future = train_test_split(daily_univariate_ts, test_days)
        forecaster = Prophet(n_changepoints=0, stan_backend=backend)
        forecaster.fit(train)
        forecaster.predict(future)
        assert forecaster.params is not None
        assert forecaster.n_changepoints == 0

    @pytest.mark.slow
    def test_fit_predict_no_changepoints_mcmc(self, daily_univariate_ts, backend):
        test_days = daily_univariate_ts.shape[0] // 2
        train, future = train_test_split(daily_univariate_ts, test_days)
        forecaster = Prophet(n_changepoints=0, mcmc_samples=100, stan_backend=backend)
        forecaster.fit(train, show_progress=False)
        forecaster.predict(future)
        assert forecaster.params is not None
        assert forecaster.n_changepoints == 0

    def test_fit_changepoint_not_in_history(self, daily_univariate_ts, backend):
        train = daily_univariate_ts[
            (daily_univariate_ts["ds"] < "2013-01-01") | (daily_univariate_ts["ds"] > "2014-01-01")
        ]
        future = pd.DataFrame({"ds": daily_univariate_ts["ds"]})
        prophet = Prophet(changepoints=["2013-06-06"], stan_backend=backend)
        forecaster = prophet
        forecaster.fit(train)
        forecaster.predict(future)
        assert forecaster.params is not None
        assert forecaster.n_changepoints == 1

    def test_fit_predict_duplicates(self, daily_univariate_ts, backend):
        """
        The underlying model should still fit successfully when there are duplicate dates in the history.
        The model essentially sees this as multiple observations for the same time value, and fits the parameters
        accordingly.
        """
        train, test = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        repeated_obs = train.copy()
        repeated_obs["y"] += 10
        train = pd.concat([train, repeated_obs])
        forecaster = Prophet(stan_backend=backend)
        forecaster.fit(train)
        forecaster.predict(test)

    def test_fit_predict_constant_history(self, daily_univariate_ts, backend):
        """
        When the training data history is constant, Prophet should predict the same value for all future dates.
        """
        for constant in [0, 20]:
            train, test = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
            train["y"] = constant
            forecaster = Prophet(stan_backend=backend)
            forecaster.fit(train)
            result = forecaster.predict(test)
            assert result["yhat"].values[-1] == constant

    def test_fit_predict_uncertainty_disabled(self, daily_univariate_ts, backend):
        test_days = daily_univariate_ts.shape[0] // 2
        train, future = train_test_split(daily_univariate_ts, test_days)
        for uncertainty in [0, False]:
            forecaster = Prophet(uncertainty_samples=uncertainty, stan_backend=backend)
            forecaster.fit(train)
            result = forecaster.predict(future)
            expected_cols = [
                "ds",
                "trend",
                "additive_terms",
                "multiplicative_terms",
                "weekly",
                "yhat",
            ]
            assert all(col in expected_cols for col in result.columns.tolist())


class TestProphetDataPrep:
    def test_setup_dataframe(self, daily_univariate_ts, backend):
        """Test that the columns 't' and 'y_scaled' are added to the dataframe."""
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        m = Prophet(stan_backend=backend)
        history = m.setup_dataframe(train, initialize_scales=True)

        assert "t" in history
        assert history["t"].min() == 0.0
        assert history["t"].max() == 1.0

        assert "y_scaled" in history
        assert history["y_scaled"].max() == 1.0

    def test_setup_dataframe_ds_column(self, daily_univariate_ts, backend):
        """Test case where 'ds' exists as an index name and column. Prophet should use the column."""
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        train.index = pd.to_datetime(["1970-01-01" for _ in range(train.shape[0])])
        train.index.rename("ds", inplace=True)
        m = Prophet(stan_backend=backend)
        m.fit(train)
        assert np.all(m.history["ds"].values == train["ds"].values)

    def test_logistic_floor(self, daily_univariate_ts, backend):
        """Test the scaling of y with logistic growth and a floor/cap."""
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        train["floor"] = 10.0
        train["cap"] = 80.0
        m = Prophet(growth="logistic", stan_backend=backend)
        m.fit(train)
        assert m.logistic_floor
        assert "floor" in m.history
        assert m.history["y_scaled"][0] == 1.0
        for col in ["y", "floor", "cap"]:
            train[col] += 10.0
        m2 = Prophet(growth="logistic", stan_backend=backend)
        m2.fit(train)
        assert m2.history["y_scaled"][0] == pytest.approx(1.0, 0.01)

    def test_logistic_floor_minmax(self, daily_univariate_ts, backend):
        """Test the scaling of y with logistic growth and a floor/cap."""
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        train["floor"] = 10.0
        train["cap"] = 80.0
        m = Prophet(growth="logistic", stan_backend=backend, scaling="minmax")
        m.fit(train)
        assert m.logistic_floor
        assert "floor" in m.history
        assert m.history["y_scaled"].min() > 0.0
        assert m.history["y_scaled"].max() < 1.0
        for col in ["y", "floor", "cap"]:
            train[col] += 10.0
        m2 = Prophet(growth="logistic", stan_backend=backend, scaling="minmax")
        m2.fit(train)
        assert m2.history["y_scaled"].min() > 0.0
        assert m2.history["y_scaled"].max() < 1.0
        # Check that the scaling is the same
        assert m2.history['y_scaled'].mean() == m.history['y_scaled'].mean()

    def test_make_future_dataframe(self, daily_univariate_ts, backend):
        train = daily_univariate_ts.head(468 // 2)
        forecaster = Prophet(stan_backend=backend)
        forecaster.fit(train)
        future = forecaster.make_future_dataframe(periods=3, freq="D", include_history=False)
        correct = pd.DatetimeIndex(["2013-04-26", "2013-04-27", "2013-04-28"])
        assert len(future) == 3
        assert np.all(future["ds"].values == correct.values)

        future = forecaster.make_future_dataframe(periods=3, freq="M", include_history=False)
        correct = pd.DatetimeIndex(["2013-04-30", "2013-05-31", "2013-06-30"])
        assert len(future) == 3
        assert np.all(future["ds"].values == correct.values)


class TestProphetTrendComponent:
    def test_invalid_growth_input(self, backend):
        msg = 'Parameter "growth" should be "linear", ' '"logistic" or "flat".'
        with pytest.raises(ValueError, match=msg):
            Prophet(growth="constant", stan_backend=backend)

    def test_growth_init(self, daily_univariate_ts, backend):
        model = Prophet(growth="logistic", stan_backend=backend)
        train = daily_univariate_ts.iloc[:468].copy()
        train["cap"] = train["y"].max()

        history = model.setup_dataframe(train, initialize_scales=True)

        k, m = model.linear_growth_init(history)
        assert k == pytest.approx(0.3055671)
        assert m == pytest.approx(0.5307511)

        k, m = model.logistic_growth_init(history)
        assert k == pytest.approx(1.507925, abs=1e-4)
        assert m == pytest.approx(-0.08167497, abs=1e-4)

        k, m = model.flat_growth_init(history)
        assert k == 0
        assert m == pytest.approx(0.49335657, abs=1e-4)

    def test_growth_init_minmax(self, daily_univariate_ts, backend):
        model = Prophet(growth="logistic", stan_backend=backend, scaling="minmax")
        train = daily_univariate_ts.iloc[:468].copy()
        train["cap"] = train["y"].max()

        history = model.setup_dataframe(train, initialize_scales=True)

        k, m = model.linear_growth_init(history)
        assert k == pytest.approx(0.4053406)
        assert m == pytest.approx(0.3775322)

        k, m = model.logistic_growth_init(history)
        assert k == pytest.approx(1.782523, abs=1e-4)
        assert m == pytest.approx(0.280521, abs=1e-4)

        k, m = model.flat_growth_init(history)
        assert k == 0
        assert m == pytest.approx(0.32792770, abs=1e-4)

    @pytest.mark.parametrize("scaling",["absmax","minmax"])
    def test_flat_growth(self, backend, scaling):
        m = Prophet(growth="flat", stan_backend=backend, scaling=scaling)
        x = np.linspace(0, 2 * np.pi, 8 * 7)
        history = pd.DataFrame(
            {
                "ds": pd.date_range(start="2020-01-01", periods=8 * 7, freq="d"),
                "y": 30 + np.sin(x * 8.0),
            }
        )
        m.fit(history)
        future = m.make_future_dataframe(10, include_history=True)
        fcst = m.predict(future)
        m_ = m.params["m"][0, 0]
        k = m.params["k"][0, 0]
        assert k == pytest.approx(0.0)
        assert fcst["trend"].unique()[0] == pytest.approx((m_ * m.y_scale) + m.y_min)
        assert np.round((m_ * m.y_scale) + m.y_min) == 30.0

    def test_piecewise_linear(self, backend):
        model = Prophet(stan_backend=backend)

        t = np.arange(11.0)
        m = 0
        k = 1.0
        deltas = np.array([0.5])
        changepoint_ts = np.array([5])

        y = model.piecewise_linear(t, deltas, k, m, changepoint_ts)
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5])
        assert (y - y_true).sum() == 0.0

        t = t[8:]
        y_true = y_true[8:]
        y = model.piecewise_linear(t, deltas, k, m, changepoint_ts)
        assert (y - y_true).sum() == 0.0

    def test_piecewise_logistic(self, backend):
        model = Prophet(stan_backend=backend)

        t = np.arange(11.0)
        cap = np.ones(11) * 10
        m = 0
        k = 1.0
        deltas = np.array([0.5])
        changepoint_ts = np.array([5])

        y = model.piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
        y_true = np.array(
            [
                5.000000,
                7.310586,
                8.807971,
                9.525741,
                9.820138,
                9.933071,
                9.984988,
                9.996646,
                9.999252,
                9.999833,
                9.999963,
            ]
        )

        t = t[8:]
        y_true = y_true[8:]
        cap = cap[8:]
        y = model.piecewise_logistic(t, cap, deltas, k, m, changepoint_ts)
        assert (y - y_true).sum() == pytest.approx(0.0, abs=1e-5)

    def test_flat_trend(self, backend):
        model = Prophet(stan_backend=backend)
        t = np.arange(11)
        m = 0.5
        y = model.flat_trend(t, m)
        y_true = np.array([0.5] * 11)
        assert (y - y_true).sum() == 0.0
        t = t[8:]
        y_true = y_true[8:]
        y = model.flat_trend(t, m)
        assert (y - y_true).sum() == 0.0

    def test_get_changepoints(self, daily_univariate_ts, backend):
        """
        By default, Prophet uses the first 80% of the history to detect changepoints.
        """
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        m = Prophet(stan_backend=backend)
        history = m.setup_dataframe(train, initialize_scales=True)
        m.history = history
        m.set_changepoints()
        cp = m.changepoints_t
        assert cp.shape[0] == m.n_changepoints
        assert len(cp.shape) == 1
        assert cp.min() > 0
        cp_indx = int(np.ceil(0.8 * history.shape[0]))
        assert cp.max() <= history["t"].values[cp_indx]

    def test_set_changepoint_range(self, daily_univariate_ts, backend):
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        m = Prophet(changepoint_range=0.4, stan_backend=backend)
        history = m.setup_dataframe(train, initialize_scales=True)
        m.history = history
        m.set_changepoints()
        cp = m.changepoints_t
        assert cp.shape[0] == m.n_changepoints
        assert len(cp.shape) == 1
        assert cp.min() > 0
        cp_indx = int(np.ceil(0.4 * history.shape[0]))
        cp.max() <= history["t"].values[cp_indx]
        for out_of_range in [-0.1, 2]:
            with pytest.raises(ValueError):
                m = Prophet(changepoint_range=out_of_range, stan_backend=backend)

    def test_get_zero_changepoints(self, daily_univariate_ts, backend):
        train, _ = train_test_split(daily_univariate_ts, daily_univariate_ts.shape[0] // 2)
        m = Prophet(n_changepoints=0, stan_backend=backend)
        history = m.setup_dataframe(train, initialize_scales=True)
        m.history = history
        m.set_changepoints()
        cp = m.changepoints_t
        assert cp.shape[0] == 1
        assert cp[0] == 0

    def test_override_n_changepoints(self, daily_univariate_ts, backend):
        train = daily_univariate_ts.head(20).copy()
        m = Prophet(n_changepoints=15, stan_backend=backend)
        history = m.setup_dataframe(train, initialize_scales=True)
        m.history = history
        m.set_changepoints()
        assert m.n_changepoints == 15
        cp = m.changepoints_t
        assert cp.shape[0] == 15


class TestProphetSeasonalComponent:
    def test_fourier_series_weekly(self, daily_univariate_ts):
        mat = Prophet.fourier_series(daily_univariate_ts["ds"], 7, 3)
        # These are from the R forecast package directly.
        true_values = np.array([0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837, -0.9009689])
        assert np.sum((mat[0] - true_values) ** 2) == pytest.approx(0.0)

    def test_fourier_series_yearly(self, daily_univariate_ts):
        mat = Prophet.fourier_series(daily_univariate_ts["ds"], 365.25, 3)
        # These are from the R forecast package directly.
        true_values = np.array(
            [0.7006152, -0.7135393, -0.9998330, 0.01827656, 0.7262249, 0.6874572]
        )
        assert np.sum((mat[0] - true_values) ** 2) == pytest.approx(0.0)

    def test_auto_weekly_seasonality(self, daily_univariate_ts, backend):
        # Should be enabled
        train = daily_univariate_ts.head(15)
        m = Prophet(stan_backend=backend)
        assert m.weekly_seasonality == "auto"
        m.fit(train)
        assert "weekly" in m.seasonalities
        assert m.seasonalities["weekly"] == {
            "period": 7,
            "fourier_order": 3,
            "prior_scale": 10.0,
            "mode": "additive",
            "condition_name": None,
        }
        # Should be disabled due to too short history
        train = daily_univariate_ts.head(9)
        m = Prophet(stan_backend=backend)
        m.fit(train)
        assert "weekly" not in m.seasonalities
        m = Prophet(weekly_seasonality=True, stan_backend=backend)
        m.fit(train)
        assert "weekly" in m.seasonalities
        # Should be False due to weekly spacing
        train = daily_univariate_ts.iloc[::7, :]
        m = Prophet(stan_backend=backend)
        m.fit(train)
        assert "weekly" not in m.seasonalities
        m = Prophet(weekly_seasonality=2, seasonality_prior_scale=3.0, stan_backend=backend)
        m.fit(daily_univariate_ts)
        assert m.seasonalities["weekly"] == {
            "period": 7,
            "fourier_order": 2,
            "prior_scale": 3.0,
            "mode": "additive",
            "condition_name": None,
        }

    def test_auto_yearly_seasonality(self, daily_univariate_ts, backend):
        # Should be enabled
        m = Prophet(stan_backend=backend)
        assert m.yearly_seasonality == "auto"
        m.fit(daily_univariate_ts)
        assert "yearly" in m.seasonalities
        assert m.seasonalities["yearly"] == {
            "period": 365.25,
            "fourier_order": 10,
            "prior_scale": 10.0,
            "mode": "additive",
            "condition_name": None,
        }
        # Should be disabled due to too short history
        train = daily_univariate_ts.head(240)
        m = Prophet(stan_backend=backend)
        m.fit(train)
        assert "yearly" not in m.seasonalities
        m = Prophet(yearly_seasonality=True, stan_backend=backend)
        m.fit(train)
        assert "yearly" in m.seasonalities
        m = Prophet(yearly_seasonality=7, seasonality_prior_scale=3.0, stan_backend=backend)
        m.fit(daily_univariate_ts)
        assert m.seasonalities["yearly"] == {
            "period": 365.25,
            "fourier_order": 7,
            "prior_scale": 3.0,
            "mode": "additive",
            "condition_name": None,
        }

    def test_auto_daily_seasonality(self, daily_univariate_ts, subdaily_univariate_ts, backend):
        # Should be enabled
        m = Prophet(stan_backend=backend)
        assert m.daily_seasonality == "auto"
        m.fit(subdaily_univariate_ts)
        assert "daily" in m.seasonalities
        assert m.seasonalities["daily"] == {
            "period": 1,
            "fourier_order": 4,
            "prior_scale": 10.0,
            "mode": "additive",
            "condition_name": None,
        }
        # Should be disabled due to too short history
        train = subdaily_univariate_ts.head(430)
        m = Prophet(stan_backend=backend)
        m.fit(train)
        assert "daily" not in m.seasonalities
        m = Prophet(daily_seasonality=True, stan_backend=backend)
        m.fit(train)
        assert "daily" in m.seasonalities
        m = Prophet(daily_seasonality=7, seasonality_prior_scale=3.0, stan_backend=backend)
        m.fit(subdaily_univariate_ts)
        assert m.seasonalities["daily"] == {
            "period": 1,
            "fourier_order": 7,
            "prior_scale": 3.0,
            "mode": "additive",
            "condition_name": None,
        }
        m = Prophet(stan_backend=backend)
        m.fit(daily_univariate_ts)
        assert "daily" not in m.seasonalities

    def test_set_seasonality_mode(self, backend):
        # Setting attribute
        m = Prophet(stan_backend=backend)
        assert m.seasonality_mode == "additive"
        m = Prophet(seasonality_mode="multiplicative", stan_backend=backend)
        assert m.seasonality_mode == "multiplicative"
        with pytest.raises(ValueError):
            Prophet(seasonality_mode="batman", stan_backend=backend)

    def test_set_holidays_mode(self, backend):
        # Setting attribute
        m = Prophet(stan_backend=backend)
        assert m.holidays_mode == "additive"
        m = Prophet(seasonality_mode="multiplicative", stan_backend=backend)
        assert m.holidays_mode == "multiplicative"
        m = Prophet(holidays_mode="multiplicative", stan_backend=backend)
        assert m.holidays_mode == "multiplicative"
        with pytest.raises(ValueError):
            Prophet(holidays_mode="batman", stan_backend=backend)

    def test_seasonality_modes(self, daily_univariate_ts, backend):
        # Model with holidays, seasonalities, and extra regressors
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2016-12-25"]),
                "holiday": ["xmas"],
                "lower_window": [-1],
                "upper_window": [0],
            }
        )
        m = Prophet(seasonality_mode="multiplicative", holidays=holidays, stan_backend=backend)
        m.add_seasonality("monthly", period=30, mode="additive", fourier_order=3)
        m.add_regressor("binary_feature", mode="additive")
        m.add_regressor("numeric_feature")
        # Construct seasonal features
        df = daily_univariate_ts.copy()
        df["binary_feature"] = [0] * 255 + [1] * 255
        df["numeric_feature"] = range(510)
        df = m.setup_dataframe(df, initialize_scales=True)
        m.history = df.copy()
        m.set_auto_seasonalities()
        seasonal_features, prior_scales, component_cols, modes = m.make_all_seasonality_features(df)
        assert sum(component_cols["additive_terms"]) == 7
        assert sum(component_cols["multiplicative_terms"]) == 29
        assert set(modes["additive"]) == {
            "monthly",
            "binary_feature",
            "additive_terms",
            "extra_regressors_additive",
        }
        assert set(modes["multiplicative"]) == {
            "weekly",
            "yearly",
            "xmas",
            "numeric_feature",
            "multiplicative_terms",
            "extra_regressors_multiplicative",
            "holidays",
        }


class TestProphetCustomSeasonalComponent:
    def test_custom_monthly_seasonality(self, backend):
        m = Prophet(stan_backend=backend)
        m.add_seasonality(name="monthly", period=30, fourier_order=5, prior_scale=2.0)
        assert m.seasonalities["monthly"] == {
            "period": 30,
            "fourier_order": 5,
            "prior_scale": 2.0,
            "mode": "additive",
            "condition_name": None,
        }

    def test_duplicate_component_names(self, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2017-01-02"]),
                "holiday": ["special_day"],
                "prior_scale": [4.0],
            }
        )
        m = Prophet(holidays=holidays, stan_backend=backend)

        with pytest.raises(ValueError):
            m.add_seasonality(name="special_day", period=30, fourier_order=5)
        with pytest.raises(ValueError):
            m.add_seasonality(name="trend", period=30, fourier_order=5)
        m.add_seasonality(name="weekly", period=30, fourier_order=5)

    def test_custom_fourier_order(self, backend):
        """Fourier order cannot be <= 0"""
        m = Prophet(stan_backend=backend)
        with pytest.raises(ValueError):
            m.add_seasonality(name="weekly", period=7, fourier_order=0)
        with pytest.raises(ValueError):
            m.add_seasonality(name="weekly", period=7, fourier_order=-1)

    def test_custom_priors(self, daily_univariate_ts, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2017-01-02"]),
                "holiday": ["special_day"],
                "prior_scale": [4.0],
            }
        )
        m = Prophet(
            holidays=holidays,
            yearly_seasonality=False,
            seasonality_mode="multiplicative",
            stan_backend=backend,
        )
        m.add_seasonality(
            name="monthly", period=30, fourier_order=5, prior_scale=2.0, mode="additive"
        )
        m.fit(daily_univariate_ts)
        assert m.seasonalities["monthly"]["mode"] == "additive"
        assert m.seasonalities["weekly"]["mode"] == "multiplicative"
        seasonal_features, prior_scales, component_cols, modes = m.make_all_seasonality_features(
            m.history
        )
        assert sum(component_cols["monthly"]) == 10
        assert sum(component_cols["special_day"]) == 1
        assert sum(component_cols["weekly"]) == 6
        assert sum(component_cols["additive_terms"]) == 10
        assert sum(component_cols["multiplicative_terms"]) == 7

        if seasonal_features.columns[0] == "monthly_delim_1":
            true = [2.0] * 10 + [10.0] * 6 + [4.0]
            assert sum(component_cols["monthly"][:10]) == 10
            assert sum(component_cols["weekly"][10:16]) == 6
        else:
            true = [10.0] * 6 + [2.0] * 10 + [4.0]
            assert sum(component_cols["weekly"][:6]) == 6
            assert sum(component_cols["monthly"][6:16]) == 10
        assert prior_scales == true

    def test_conditional_custom_seasonality(self, daily_univariate_ts, backend):
        m = Prophet(weekly_seasonality=False, yearly_seasonality=False, stan_backend=backend)
        m.add_seasonality(
            name="conditional_weekly",
            period=7,
            fourier_order=3,
            prior_scale=2.0,
            condition_name="is_conditional_week",
        )
        m.add_seasonality(name="normal_monthly", period=30.5, fourier_order=5, prior_scale=2.0)
        df = daily_univariate_ts.copy()
        with pytest.raises(ValueError):
            # Require all conditions names in df
            m.fit(df)
        df["is_conditional_week"] = [0] * 255 + [2] * 255
        with pytest.raises(ValueError):
            # Require boolean compatible values
            m.fit(df)
        df["is_conditional_week"] = [0] * 255 + [1] * 255
        m.fit(df)
        assert m.seasonalities["conditional_weekly"] == {
            "period": 7,
            "fourier_order": 3,
            "prior_scale": 2.0,
            "mode": "additive",
            "condition_name": "is_conditional_week",
        }
        assert m.seasonalities["normal_monthly"]["condition_name"] is None
        seasonal_features, prior_scales, component_cols, modes = m.make_all_seasonality_features(
            m.history
        )
        # Confirm that only values without is_conditional_week has non zero entries
        condition_cols = [
            c for c in seasonal_features.columns if c.startswith("conditional_weekly")
        ]
        assert np.array_equal(
            (seasonal_features[condition_cols] != 0).any(axis=1).values,
            df["is_conditional_week"].values,
        )


class TestProphetHolidays:
    def test_holidays_lower_window(self, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2016-12-25"]),
                "holiday": ["xmas"],
                "lower_window": [-1],
                "upper_window": [0],
            }
        )
        model = Prophet(holidays=holidays, stan_backend=backend)
        df = pd.DataFrame({"ds": pd.date_range("2016-12-20", "2016-12-31")})
        feats, priors, names = model.make_holiday_features(df["ds"], model.holidays)
        assert feats.shape == (df.shape[0], 2)
        assert (feats.sum(axis=0) - np.array([1.0, 1.0])).sum() == 0.0
        assert priors == [10.0, 10.0]  # Default prior
        assert names == ["xmas"]

    def test_holidays_upper_window(self, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2016-12-25"]),
                "holiday": ["xmas"],
                "lower_window": [-1],
                "upper_window": [10],
            }
        )
        m = Prophet(holidays=holidays, stan_backend=backend)
        df = pd.DataFrame({"ds": pd.date_range("2016-12-20", "2016-12-31")})
        feats, priors, names = m.make_holiday_features(df["ds"], m.holidays)
        # 12 columns generated even though only 8 overlap
        assert feats.shape == (df.shape[0], 12)
        assert priors == [10.0 for _ in range(12)]
        assert names == ["xmas"]

    def test_holidays_priors(self, backend):
        # Check prior specifications
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2016-12-25", "2017-12-25"]),
                "holiday": ["xmas", "xmas"],
                "lower_window": [-1, -1],
                "upper_window": [0, 0],
                "prior_scale": [5.0, 5.0],
            }
        )
        m = Prophet(holidays=holidays, stan_backend=backend)
        df = pd.DataFrame({"ds": pd.date_range("2016-12-20", "2016-12-31")})
        feats, priors, names = m.make_holiday_features(df["ds"], m.holidays)
        assert priors == [5.0, 5.0]
        assert names == ["xmas"]
        # 2 different priors
        holidays2 = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2012-06-06", "2013-06-06"]),
                "holiday": ["seans-bday"] * 2,
                "lower_window": [0] * 2,
                "upper_window": [1] * 2,
                "prior_scale": [8] * 2,
            }
        )
        holidays2 = pd.concat((holidays, holidays2), sort=True)
        m = Prophet(holidays=holidays2, stan_backend=backend)
        feats, priors, names = m.make_holiday_features(df["ds"], m.holidays)
        pn = zip(priors, [s.split("_delim_")[0] for s in feats.columns])
        for t in pn:
            assert t in [(8.0, "seans-bday"), (5.0, "xmas")]
        holidays2 = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2012-06-06", "2013-06-06"]),
                "holiday": ["seans-bday"] * 2,
                "lower_window": [0] * 2,
                "upper_window": [1] * 2,
            }
        )
        holidays2 = pd.concat((holidays, holidays2), sort=True)
        feats, priors, names = Prophet(
            holidays=holidays2, holidays_prior_scale=4, stan_backend=backend
        ).make_holiday_features(df["ds"], holidays2)
        assert set(priors) == {4.0, 5.0}

    def test_holidays_bad_priors(self, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2016-12-25", "2016-12-27"]),
                "holiday": ["xmasish", "xmasish"],
                "lower_window": [-1, -1],
                "upper_window": [0, 0],
                "prior_scale": [5.0, 6.0],
            }
        )
        df = pd.DataFrame({"ds": pd.date_range("2016-12-20", "2016-12-31")})
        with pytest.raises(ValueError):
            Prophet(holidays=holidays, stan_backend=backend).make_holiday_features(
                df["ds"], holidays
            )

    def test_fit_with_holidays(self, daily_univariate_ts, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2012-06-06", "2013-06-06"]),
                "holiday": ["seans-bday"] * 2,
                "lower_window": [0] * 2,
                "upper_window": [1] * 2,
            }
        )
        model = Prophet(holidays=holidays, uncertainty_samples=0, stan_backend=backend)
        model.fit(daily_univariate_ts).predict()

    def test_fit_predict_with_country_holidays(self, daily_univariate_ts, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2012-06-06", "2013-06-06"]),
                "holiday": ["seans-bday"] * 2,
                "lower_window": [0] * 2,
                "upper_window": [1] * 2,
            }
        )
        # Test with holidays and country_holidays
        model = Prophet(holidays=holidays, uncertainty_samples=0, stan_backend=backend)
        model.add_country_holidays(country_name="US")
        model.fit(daily_univariate_ts).predict()
        # There are training holidays missing in the test set
        train = daily_univariate_ts.head(154)
        future = daily_univariate_ts.tail(355)
        model = Prophet(uncertainty_samples=0, stan_backend=backend)
        model.add_country_holidays(country_name="US")
        model.fit(train).predict(future)
        # There are test holidays missing in the training set
        train = daily_univariate_ts.tail(355)
        model = Prophet(uncertainty_samples=0, stan_backend=backend)
        model.add_country_holidays(country_name="US")
        model.fit(train)
        future = model.make_future_dataframe(periods=60, include_history=False)
        model.predict(future)

    def test_subdaily_holidays(self, subdaily_univariate_ts, backend):
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2017-01-02"]),
                "holiday": ["special_day"],
            }
        )
        m = Prophet(holidays=holidays, stan_backend=backend)
        m.fit(subdaily_univariate_ts)
        fcst = m.predict()
        assert sum(fcst["special_day"] == 0) == 575



class TestProphetRegressors:
    def test_added_regressors(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend)
        m.add_regressor("binary_feature", prior_scale=0.2)
        m.add_regressor("numeric_feature", prior_scale=0.5)
        m.add_regressor("numeric_feature2", prior_scale=0.5, mode="multiplicative")
        m.add_regressor("binary_feature2", standardize=True)
        df = daily_univariate_ts.copy()
        df["binary_feature"] = ["0"] * 255 + ["1"] * 255
        df["numeric_feature"] = range(510)
        df["numeric_feature2"] = range(510)
        with pytest.raises(ValueError):
            # Require all regressors in df
            m.fit(df)
        df["binary_feature2"] = [1] * 100 + [0] * 410
        m.fit(df)
        # Check that standardizations are correctly set
        assert m.extra_regressors["binary_feature"] == {
            "prior_scale": 0.2,
            "mu": 0,
            "std": 1,
            "standardize": "auto",
            "mode": "additive",
        }
        assert m.extra_regressors["numeric_feature"]["prior_scale"] == 0.5
        assert m.extra_regressors["numeric_feature"]["mu"] == 254.5
        assert m.extra_regressors["numeric_feature"]["std"] == pytest.approx(147.368585, abs=1e-5)
        assert m.extra_regressors["numeric_feature2"]["mode"] == "multiplicative"
        assert m.extra_regressors["binary_feature2"]["prior_scale"] == 10.0
        assert m.extra_regressors["binary_feature2"]["mu"] == pytest.approx(0.1960784, abs=1e-5)
        assert m.extra_regressors["binary_feature2"]["std"] == pytest.approx(0.3974183, abs=1e-5)
        # Check that standardization is done correctly
        df2 = m.setup_dataframe(df.copy())
        assert df2["binary_feature"][0] == 0
        assert df2["numeric_feature"][0] == pytest.approx(-1.726962, abs=1e-4)
        assert df2["binary_feature2"][0] == pytest.approx(2.022859, abs=1e-4)
        # Check that feature matrix and prior scales are correctly constructed
        seasonal_features, prior_scales, component_cols, modes = m.make_all_seasonality_features(
            df2
        )
        assert seasonal_features.shape[1] == 30
        names = ["binary_feature", "numeric_feature", "binary_feature2"]
        true_priors = [0.2, 0.5, 10.0]
        for i, name in enumerate(names):
            assert name in seasonal_features
            assert sum(component_cols[name]) == 1
            assert sum(np.array(prior_scales) * component_cols[name]) == true_priors[i]
        # Check that forecast components are reasonable
        future = pd.DataFrame(
            {
                "ds": ["2014-06-01"],
                "binary_feature": [0],
                "numeric_feature": [10],
                "numeric_feature2": [10],
            }
        )
        # future dataframe also requires regressor values
        with pytest.raises(ValueError):
            m.predict(future)
        future["binary_feature2"] = 0
        fcst = m.predict(future)
        assert fcst.shape[1] == 37
        assert fcst["binary_feature"][0] == 0
        assert fcst["extra_regressors_additive"][0] == pytest.approx(
            fcst["numeric_feature"][0] + fcst["binary_feature2"][0]
        )
        assert fcst["extra_regressors_multiplicative"][0] == pytest.approx(
            fcst["numeric_feature2"][0]
        )
        assert fcst["additive_terms"][0] == pytest.approx(
            fcst["yearly"][0] + fcst["weekly"][0] + fcst["extra_regressors_additive"][0]
        )
        assert fcst["multiplicative_terms"][0] == pytest.approx(
            fcst["extra_regressors_multiplicative"][0]
        )
        assert fcst["yhat"][0] == pytest.approx(
            fcst["trend"][0] * (1 + fcst["multiplicative_terms"][0]) + fcst["additive_terms"][0]
        )

    def test_constant_regressor(self, daily_univariate_ts, backend):
        df = daily_univariate_ts.copy()
        df["constant_feature"] = 0
        m = Prophet(stan_backend=backend)
        m.add_regressor("constant_feature")
        m.fit(df)
        assert m.extra_regressors["constant_feature"]["std"] == 1


class TestProphetWarmStart:
    def test_fit_warm_start(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend).fit(daily_univariate_ts.iloc[:500])
        m2 = Prophet(stan_backend=backend).fit(
            daily_univariate_ts.iloc[:510], init=warm_start_params(m)
        )
        assert len(m2.params["delta"][0]) == 25

    def test_sampling_warm_start(self, daily_univariate_ts, backend):
        m = Prophet(mcmc_samples=100, stan_backend=backend).fit(
            daily_univariate_ts.iloc[:500], show_progress=False
        )
        m2 = Prophet(mcmc_samples=100, stan_backend=backend).fit(
            daily_univariate_ts.iloc[:510], init=warm_start_params(m), show_progress=False
        )
        assert m2.params["delta"].shape == (200, 25)
