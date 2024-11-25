# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from prophet import Prophet
from prophet.utilities import regressor_coefficients

from prophet.utilities import warm_start_params

class TestUtilities:
    def test_regressor_coefficients(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend)
        df = daily_univariate_ts.copy()
        np.random.seed(123)
        df["regr1"] = np.random.normal(size=df.shape[0])
        df["regr2"] = np.random.normal(size=df.shape[0])
        m.add_regressor("regr1", mode="additive")
        m.add_regressor("regr2", mode="multiplicative")
        m.fit(df)

        coefs = regressor_coefficients(m)
        assert coefs.shape == (2, 6)
        # No MCMC sampling, so lower and upper should be the same as mean
        assert np.array_equal(coefs["coef_lower"].values, coefs["coef"].values)
        assert np.array_equal(coefs["coef_upper"].values, coefs["coef"].values)

    def test_warm_start_params_with_mcmc(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend, mcmc_samples=10)
        df = daily_univariate_ts.copy()
        np.random.seed(123)
        df["regr1"] = np.random.normal(size=df.shape[0])
        m.add_regressor("regr1", mode="additive")
        m.fit(df)
    
        params = warm_start_params(m)
        assert isinstance(params, dict)
        assert "k" in params
        assert "m" in params
        assert "sigma_obs" in params
        assert "delta" in params
        assert "beta" in params


    def test_warm_start_params_no_mcmc(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend, mcmc_samples=0)
        df = daily_univariate_ts.copy()
        np.random.seed(123)
        df["regr1"] = np.random.normal(size=df.shape[0])
        m.add_regressor("regr1", mode="additive")
        m.fit(df)
    
        params = warm_start_params(m)
        assert isinstance(params, dict)
        assert "k" in params
        assert "m" in params
        assert "sigma_obs" in params
        assert "delta" in params
        assert "beta" in params

