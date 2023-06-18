# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prophet import Prophet
from prophet.serialize import PD_DATAFRAME, PD_SERIES, model_from_json, model_to_json


class TestSerialize:
    def test_simple_serialize(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend)
        df = daily_univariate_ts.head(daily_univariate_ts.shape[0] - 30)
        m.fit(df)

        future = m.make_future_dataframe(2, include_history=False)
        fcst = m.predict(future)

        model_str = model_to_json(m)
        # Make sure json doesn't get too large in the future
        assert len(model_str) < 200000
        m2 = model_from_json(model_str)

        # Check that m and m2 are equal
        assert m.__dict__.keys() == m2.__dict__.keys()
        for k, v in m.__dict__.items():
            if k in ["stan_fit", "stan_backend"]:
                continue
            if k == "params":
                assert v.keys() == m2.params.keys()
                for kk, vv in v.items():
                    assert np.array_equal(vv, m2.params[kk])
            elif k in PD_SERIES and v is not None:
                assert v.equals(m2.__dict__[k])
            elif k in PD_DATAFRAME and v is not None:
                pd.testing.assert_frame_equal(v, m2.__dict__[k], check_index_type=False)
            elif k == "changepoints_t":
                assert np.array_equal(v, m.__dict__[k])
            else:
                assert v == m2.__dict__[k]
        assert m2.stan_fit is None
        assert m2.stan_backend is None

        # Check that m2 makes the same forecast
        future2 = m2.make_future_dataframe(2, include_history=False)
        fcst2 = m2.predict(future2)

        assert np.array_equal(fcst["yhat"].values, fcst2["yhat"].values)

    def test_full_serialize(self, daily_univariate_ts, backend):
        # Construct a model with all attributes
        holidays = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2012-06-06", "2013-06-06"]),
                "holiday": ["seans-bday"] * 2,
                "lower_window": [0] * 2,
                "upper_window": [1] * 2,
            }
        )
        # Test with holidays and country_holidays
        m = Prophet(
            holidays=holidays,
            seasonality_mode="multiplicative",
            changepoints=["2012-07-01", "2012-10-01", "2013-01-01"],
            stan_backend=backend,
        )
        m.add_country_holidays(country_name="US")
        m.add_seasonality(
            name="conditional_weekly",
            period=7,
            fourier_order=3,
            prior_scale=2.0,
            condition_name="is_conditional_week",
        )
        m.add_seasonality(name="normal_monthly", period=30.5, fourier_order=5, prior_scale=2.0)
        df = daily_univariate_ts.copy()
        df["is_conditional_week"] = [0] * 255 + [1] * 255
        m.add_regressor("binary_feature", prior_scale=0.2)
        m.add_regressor("numeric_feature", prior_scale=0.5)
        m.add_regressor("numeric_feature2", prior_scale=0.5, mode="multiplicative")
        m.add_regressor("binary_feature2", standardize=True)
        df["binary_feature"] = ["0"] * 255 + ["1"] * 255
        df["numeric_feature"] = range(510)
        df["numeric_feature2"] = range(510)
        df["binary_feature2"] = [1] * 100 + [0] * 410

        train = df.head(400)
        test = df.tail(100)

        m.fit(train)
        fcst = m.predict(test)
        # Serialize!
        m2 = model_from_json(model_to_json(m))

        # Check that m and m2 are equal
        assert m.__dict__.keys() == m2.__dict__.keys()
        for k, v in m.__dict__.items():
            if k in ["stan_fit", "stan_backend"]:
                continue
            if k == "params":
                assert v.keys() == m2.params.keys()
                for kk, vv in v.items():
                    assert np.array_equal(vv, m2.params[kk])
            elif k in PD_SERIES and v is not None:
                assert v.equals(m2.__dict__[k])
            elif k in PD_DATAFRAME and v is not None:
                pd.testing.assert_frame_equal(v, m2.__dict__[k], check_index_type=False)
            elif k == "changepoints_t":
                assert np.array_equal(v, m.__dict__[k])
            else:
                assert v == m2.__dict__[k]
        assert m2.stan_fit is None
        assert m2.stan_backend is None

        # Check that m2 makes the same forecast
        fcst2 = m2.predict(test)
        assert np.array_equal(fcst["yhat"].values, fcst2["yhat"].values)

    def test_backwards_compatibility(self):
        old_versions = {
            "0.6.1.dev0": (29.3669923968994, "fb"),
            "0.7.1": (29.282810844704414, "fb"),
            "1.0.1": (29.282810844704414, ""),
        }
        for v, (pred_val, v_str) in old_versions.items():
            fname = Path(__file__).parent / f"serialized_model_v{v}.json"
            with open(fname, "r") as fin:
                model_str = json.load(fin)
            # Check that deserializes
            m = model_from_json(model_str)
            assert json.loads(model_str)[f"__{v_str}prophet_version"] == v
            # Predict
            future = m.make_future_dataframe(10)
            fcst = m.predict(future)
            assert fcst["yhat"].values[-1] == pytest.approx(pred_val)
