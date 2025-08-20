# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from prophet import Prophet
from prophet.utilities import regressor_coefficients


class TestUtilities:
    def test_regressor_coefficients(self, daily_univariate_ts, backend):
        m = Prophet(stan_backend=backend)
        df = daily_univariate_ts.copy()
        # Set fixed seed for reproducibility
        np.random.seed(123)
        # Generate random values once and assign to both columns
        random_values = np.random.normal(size=df.shape[0])
        df["regr1"] = random_values
        # Reuse same random generator for second column to avoid another call
        df["regr2"] = np.random.normal(size=df.shape[0])
        m.add_regressor("regr1", mode="additive")
        m.add_regressor("regr2", mode="multiplicative")
        m.fit(df)

        # Get coefficients once and use for all assertions
        coefs = regressor_coefficients(m)
        assert coefs.shape == (2, 6)
        # Use numpy array comparison without redundant function calls
        coef_values = coefs["coef"].values
        assert np.array_equal(coefs["coef_lower"].values, coef_values)
        assert np.array_equal(coefs["coef_upper"].values, coef_values)
