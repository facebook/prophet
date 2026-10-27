# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from prophet import Prophet
from prophet.plot import (
    _forecast_component_plot_data,
    _seasonality_plot_data,
    get_forecast_component_plotly_props,
    get_seasonality_plotly_props,
    plot_forecast_component,
    plot_seasonality,
    plot_weekly,
    plot_yearly,
    seasonality_plot_df,
)


HOLIDAYS = pd.DataFrame({
    "holiday": "playoff",
    "ds": pd.to_datetime(["2012-07-04", "2013-01-12", "2013-07-04", "2014-01-12"]),
    "lower_window": 0,
    "upper_window": 1,
})
SEASONALITIES = ["weekly", "yearly", "monthly", "weekly_on_season"]


def _add_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x"] = np.sin(np.arange(len(df)))
    df["on_season"] = df["ds"].dt.month.isin([6, 7, 8])
    return df


def _forecast_components(fcst: pd.DataFrame) -> list[str]:
    components = [
        "trend", "holidays", "extra_regressors_additive", "extra_regressors_multiplicative",
    ]
    return [name for name in components if name in fcst]


def _fit_predict(ts: pd.DataFrame, backend: str, kind: str) -> tuple[Prophet, pd.DataFrame]:
    """Fit a model with holidays, a regressor, and custom and conditional seasonalities."""
    history = _add_columns(ts)
    kwargs = {}
    if kind == "multiplicative_logistic":
        kwargs = {"growth": "logistic", "seasonality_mode": "multiplicative"}
        history["cap"] = history["y"].max() * 1.5
        history["floor"] = history["y"].min() - 1.0
    elif kind == "no_uncertainty":
        kwargs = {"uncertainty_samples": 0}
    m = Prophet(stan_backend=backend, holidays=HOLIDAYS, **kwargs)
    m.add_regressor("x")
    m.add_seasonality("monthly", period=30.5, fourier_order=3)
    m.add_seasonality("weekly_on_season", period=7, fourier_order=3, condition_name="on_season")
    m.fit(history, seed=1237861298)
    future = _add_columns(m.make_future_dataframe(periods=30))
    for col in ["cap", "floor"]:
        if col in history:
            future[col] = history[col].iloc[0]
    np.random.seed(876543987)
    return m, m.predict(future)


@pytest.fixture(params=["additive", "multiplicative_logistic"])
def model_and_forecast(request, daily_univariate_ts, backend):
    return _fit_predict(daily_univariate_ts, backend, request.param)


class TestForecastComponentPlotData:
    def test_columns(self, model_and_forecast):
        m, fcst = model_and_forecast
        # trend, holidays and the extra regressors in the model's mode
        assert len(_forecast_components(fcst)) == 3
        for name in _forecast_components(fcst):
            data = _forecast_component_plot_data(m, fcst, name)
            pd.testing.assert_frame_equal(
                data, fcst[["ds", name, name + "_lower", name + "_upper"]]
            )
            data = _forecast_component_plot_data(m, fcst, name, uncertainty=False)
            pd.testing.assert_frame_equal(data, fcst[["ds", name]])

    def test_cap_and_floor(self, daily_univariate_ts, backend):
        m, fcst = _fit_predict(daily_univariate_ts, backend, "multiplicative_logistic")
        data = _forecast_component_plot_data(m, fcst, "trend", plot_cap=True)
        assert list(data.columns) == ["ds", "trend", "trend_lower", "trend_upper", "cap", "floor"]
        data = _forecast_component_plot_data(m, fcst, "trend", plot_cap=False)
        assert list(data.columns) == ["ds", "trend", "trend_lower", "trend_upper"]

    def test_without_uncertainty_samples(self, daily_univariate_ts, backend):
        m, fcst = _fit_predict(daily_univariate_ts, backend, "no_uncertainty")
        assert "trend_lower" not in fcst
        for name in _forecast_components(fcst):
            data = _forecast_component_plot_data(m, fcst, name, uncertainty=True)
            assert list(data.columns) == ["ds", name]

    @pytest.mark.parametrize("uncertainty", [True, False])
    def test_matplotlib_draws_data(self, model_and_forecast, uncertainty):
        m, fcst = model_and_forecast
        for name in _forecast_components(fcst):
            data = _forecast_component_plot_data(m, fcst, name, uncertainty=uncertainty)
            ax = Figure().add_subplot(111)
            artists = plot_forecast_component(m, fcst, name, ax=ax, uncertainty=uncertainty)
            np.testing.assert_array_equal(artists[0].get_xdata(), data["ds"])
            np.testing.assert_array_equal(artists[0].get_ydata(), data[name])
            # The line, plus the uncertainty interval when there is one.
            assert len(artists) == (2 if uncertainty else 1)

    def test_plotly_draws_data(self, model_and_forecast):
        pytest.importorskip("plotly")
        m, fcst = model_and_forecast
        for name in _forecast_components(fcst):
            data = _forecast_component_plot_data(m, fcst, name, plot_cap=True)
            props = get_forecast_component_plotly_props(m, fcst, name, plot_cap=True)
            traces = {trace.name: trace for trace in props["traces"]}
            np.testing.assert_array_equal(pd.to_datetime(traces[name].x), data["ds"])
            np.testing.assert_array_equal(traces[name].y, data[name])
            # Plotly leaves out the interval when it is empty.
            has_interval = (data[name + "_upper"] != data[name + "_lower"]).any()
            assert (name + "_upper" in traces) == has_interval
            assert ("Cap" in traces) == ("cap" in data)
            assert ("Floor" in traces) == ("floor" in data)
            for col, trace_name in [
                (name + "_upper", name + "_upper"),
                (name + "_lower", name + "_lower"),
                ("cap", "Cap"),
                ("floor", "Floor"),
            ]:
                if trace_name in traces:
                    np.testing.assert_array_equal(traces[trace_name].y, data[col])


class TestSeasonalityPlotData:
    def test_matches_seasonal_components(self, model_and_forecast):
        m, _ = model_and_forecast
        days = pd.date_range("2017-01-01", periods=30)
        expected = m.predict_seasonal_components(seasonality_plot_df(m, days))
        for name in SEASONALITIES:
            data = _seasonality_plot_data(m, name, days)
            assert list(data.columns) == ["ds", name, name + "_lower", name + "_upper"]
            np.testing.assert_array_equal(data["ds"], days)
            for col in [name, name + "_lower", name + "_upper"]:
                np.testing.assert_array_equal(data[col], expected[col])
            data = _seasonality_plot_data(m, name, days, uncertainty=False)
            assert list(data.columns) == ["ds", name]

    def test_conditional_seasonality_is_active(self, daily_univariate_ts, backend):
        m, _ = _fit_predict(daily_univariate_ts, backend, "additive")
        days = pd.date_range("2017-01-01", periods=7)
        data = _seasonality_plot_data(m, "weekly_on_season", days)
        assert (data["weekly_on_season"] != 0).any()

    def test_without_uncertainty_samples(self, daily_univariate_ts, backend):
        m, _ = _fit_predict(daily_univariate_ts, backend, "no_uncertainty")
        days = pd.date_range("2017-01-01", periods=7)
        data = _seasonality_plot_data(m, "weekly", days, uncertainty=True)
        assert list(data.columns) == ["ds", "weekly"]

    def test_logistic_minmax_without_floor(self, daily_univariate_ts, backend):
        # With minmax scaling and no floor column, the floor is the minimum of y.
        history = daily_univariate_ts.copy()
        history["cap"] = history["y"].max() * 1.5
        assert history["y"].min() > 1
        m = Prophet(stan_backend=backend, growth="logistic", scaling="minmax")
        m.fit(history, seed=1237861298)
        days = pd.date_range("2017-01-01", periods=7)
        data = _seasonality_plot_data(m, "weekly", days)
        fcst = m.predict(pd.DataFrame({"ds": days, "cap": history["cap"].iloc[0]}))
        for col in ["weekly", "weekly_lower", "weekly_upper"]:
            np.testing.assert_allclose(data[col], fcst[col])

    @pytest.mark.parametrize("uncertainty", [True, False])
    def test_matplotlib_draws_data(self, model_and_forecast, uncertainty):
        m, _ = model_and_forecast
        n_artists = 2 if uncertainty else 1

        days = pd.date_range("2017-01-01", periods=7) + pd.Timedelta(days=2)
        data = _seasonality_plot_data(m, "weekly_on_season", days)
        artists = plot_weekly(m, ax=Figure().add_subplot(111), uncertainty=uncertainty,
                              weekly_start=2, name="weekly_on_season")
        np.testing.assert_array_equal(artists[0].get_ydata(), data["weekly_on_season"])
        assert len(artists) == n_artists

        days = pd.date_range("2017-01-01", periods=365) + pd.Timedelta(days=10)
        data = _seasonality_plot_data(m, "yearly", days)
        artists = plot_yearly(m, ax=Figure().add_subplot(111), uncertainty=uncertainty,
                              yearly_start=10)
        np.testing.assert_array_equal(artists[0].get_xdata(), data["ds"])
        np.testing.assert_array_equal(artists[0].get_ydata(), data["yearly"])
        assert len(artists) == n_artists

        start = pd.Timestamp("2017-01-01")
        end = start + pd.Timedelta(days=30.5)
        days = pd.to_datetime(np.linspace(start.value, end.value, 200))
        data = _seasonality_plot_data(m, "monthly", days)
        artists = plot_seasonality(m, "monthly", ax=Figure().add_subplot(111),
                                   uncertainty=uncertainty)
        np.testing.assert_array_equal(artists[0].get_xdata(), data["ds"])
        np.testing.assert_array_equal(artists[0].get_ydata(), data["monthly"])
        assert len(artists) == n_artists

    def test_plotly_draws_data(self, model_and_forecast):
        pytest.importorskip("plotly")
        m, _ = model_and_forecast
        # The history is daily, so the weekly plot has one point per day.
        days = pd.date_range("2017-01-01", periods=7)
        data = _seasonality_plot_data(m, "weekly", days)
        traces = get_seasonality_plotly_props(m, "weekly")["traces"]
        np.testing.assert_array_equal(pd.to_datetime(traces[0].x), data["ds"])
        np.testing.assert_array_equal(traces[0].y, data["weekly"])
