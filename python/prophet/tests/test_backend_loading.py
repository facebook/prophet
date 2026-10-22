# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
from unittest.mock import patch, MagicMock
import pytest

import numpy as np
import pandas as pd

from prophet import Prophet
from prophet.models import CmdStanPyBackend, ModelInputData


class TestWindowsTbbPath:
    """Test that bundled TBB directory is prepended to PATH on Windows."""

    def test_cmdstanpy_backend_prepends_tbb_on_windows(self):
        """Test that CmdStanPyBackend prepends TBB to PATH once on Windows."""
        expected_tbb_dir = f"/fake/prophet/stan_model/cmdstan-{CmdStanPyBackend.CMDSTAN_VERSION}/stan/lib/stan_math/lib/tbb"

        class MockResource:
            def __init__(self, path):
                self.path = path

            def __truediv__(self, other):
                return MockResource(self.path + "/" + str(other))

            def __str__(self):
                return self.path

            def exists(self):
                return True

        mock_cmdstanpy = MagicMock()
        mock_cmdstanpy.set_cmdstan_path = MagicMock()

        with patch('prophet.models.os.name', 'nt'):
            with patch.dict(os.environ, {'PATH': '/usr/bin:/bin'}):
                with patch('prophet.models.importlib.resources.files') as mock_files:
                    mock_files.return_value = MockResource("/fake/prophet")
                    with patch.dict('sys.modules', {'cmdstanpy': mock_cmdstanpy}):
                        with patch.object(CmdStanPyBackend, 'load_model', return_value=MagicMock()):
                            # First construction should add TBB to PATH
                            CmdStanPyBackend()
                            # Second construction should NOT duplicate it
                            CmdStanPyBackend()

                # Verify TBB directory appears exactly once in PATH
                path_entries = os.environ['PATH'].split(os.pathsep)
                tbb_count = path_entries.count(expected_tbb_dir)
                assert tbb_count == 1, f"TBB directory should appear exactly once, found {tbb_count} times"
                assert path_entries[0] == expected_tbb_dir, "TBB directory should be first in PATH"


class TestBackendLoadError:
    """Test that backend load errors are properly raised with chained exceptions."""

    def test_backend_load_error_raises_runtime_error(self):
        """When backend loading fails, should raise RuntimeError with __cause__ set."""

        # Mock StanBackendEnum.get_backend_class to raise an exception
        from prophet.forecaster import StanBackendEnum

        test_exception = ValueError("Simulated backend load failure")

        with patch.object(StanBackendEnum, 'get_backend_class') as mock_get_backend:
            mock_get_backend.side_effect = test_exception

            with pytest.raises(RuntimeError) as exc_info:
                Prophet(stan_backend='CMDSTANPY')

            # Verify the RuntimeError was raised with the original exception as __cause__
            assert exc_info.value.__cause__ is test_exception, "RuntimeError should be chained from the original exception"
            assert "backend" in str(exc_info.value).lower(), "RuntimeError message should mention backend"

    def test_backend_auto_load_error_raises_runtime_error(self):
        """When auto-loading backends and all fail, should raise RuntimeError with __cause__ set."""

        from prophet.forecaster import StanBackendEnum

        test_exception = ValueError("Simulated backend load failure")

        with patch.object(StanBackendEnum, 'get_backend_class') as mock_get_backend:
            mock_get_backend.side_effect = test_exception

            with pytest.raises(RuntimeError) as exc_info:
                Prophet(stan_backend=None)

            # Verify the RuntimeError was raised with a chained exception
            assert exc_info.value.__cause__ is not None, "RuntimeError should have __cause__ set"
            assert isinstance(exc_info.value.__cause__, Exception), "RuntimeError.__cause__ should be an Exception"
            assert "backend" in str(exc_info.value).lower(), "RuntimeError message should mention backend"

            # Verify the exception chain leads to the original error
            current = exc_info.value
            found_original = False
            while current is not None:
                if current is test_exception:
                    found_original = True
                    break
                current = current.__cause__
            assert found_original, "Exception chain should contain the original exception"


class TestSanitizeCustomInits:
    """`fit(init=...)` accepts partial and list-valued init dicts, as cmdstanpy does."""

    n_changepoints = 4
    n_regressors = 2

    def default_inits(self):
        return {
            'k': 0.5,
            'm': 1.0,
            'sigma_obs': 1.0,
            'delta': np.zeros(self.n_changepoints),
            'beta': np.zeros(self.n_regressors),
        }

    def test_ndarray_init_is_kept(self):
        custom = {
            'k': 0.25,
            'm': 2.0,
            'sigma_obs': 0.5,
            'delta': np.arange(self.n_changepoints, dtype=float),
            'beta': np.arange(self.n_regressors, dtype=float),
        }
        res = CmdStanPyBackend.sanitize_custom_inits(self.default_inits(), custom)
        assert res['k'] == 0.25
        assert res['m'] == 2.0
        assert res['sigma_obs'] == 0.5
        np.testing.assert_array_equal(res['delta'], custom['delta'])
        np.testing.assert_array_equal(res['beta'], custom['beta'])

    def test_list_init_is_kept(self):
        custom = {
            'k': 0.25,
            'm': 2.0,
            'sigma_obs': 0.5,
            'delta': [0.1, 0.2, 0.3, 0.4],
            'beta': [1.5, 2.5],
        }
        res = CmdStanPyBackend.sanitize_custom_inits(self.default_inits(), custom)
        np.testing.assert_array_equal(res['delta'], np.array(custom['delta']))
        np.testing.assert_array_equal(res['beta'], np.array(custom['beta']))

    def test_missing_params_fall_back_to_defaults(self):
        res = CmdStanPyBackend.sanitize_custom_inits(
            self.default_inits(), {'k': 0.25, 'beta': [1.0, 2.0]}
        )
        assert res['k'] == 0.25
        assert res['m'] == 1.0
        assert res['sigma_obs'] == 1.0
        np.testing.assert_array_equal(res['delta'], np.zeros(self.n_changepoints))
        np.testing.assert_array_equal(res['beta'], np.array([1.0, 2.0]))

    def test_wrong_shape_falls_back_to_defaults(self):
        res = CmdStanPyBackend.sanitize_custom_inits(
            self.default_inits(), {'delta': [0.1, 0.2], 'beta': [1.0, 2.0, 3.0, 4.0]}
        )
        np.testing.assert_array_equal(res['delta'], np.zeros(self.n_changepoints))
        np.testing.assert_array_equal(res['beta'], np.zeros(self.n_regressors))

    def test_unusable_values_fall_back_to_defaults(self):
        res = CmdStanPyBackend.sanitize_custom_inits(
            self.default_inits(), {'delta': [[0.1, 0.2], [0.3]], 'beta': None}
        )
        np.testing.assert_array_equal(res['delta'], np.zeros(self.n_changepoints))
        np.testing.assert_array_equal(res['beta'], np.zeros(self.n_regressors))

    def test_sanitized_init_feeds_prepare_data(self):
        """`prepare_data` writes the init file, so it needs every key as an array-like."""
        res = CmdStanPyBackend.sanitize_custom_inits(
            self.default_inits(), {'delta': [0.1] * 4, 'beta': [1.0, 2.0]}
        )
        assert set(res) == {'k', 'm', 'sigma_obs', 'delta', 'beta'}
        for param in ['delta', 'beta']:
            assert res[param].ndim == 1

        data = dataclasses.asdict(
            ModelInputData(
                T=3,
                S=self.n_changepoints,
                K=self.n_regressors,
                tau=0.05,
                trend_indicator=0,
                y=np.zeros(3),
                t=np.zeros(3),
                cap=np.zeros(3),
                t_change=np.zeros(self.n_changepoints),
                s_a=np.zeros(self.n_regressors, dtype=int),
                s_m=np.zeros(self.n_regressors, dtype=int),
                X=pd.DataFrame(np.zeros((3, self.n_regressors))),
                sigmas=[0.05] * self.n_regressors,
            )
        )
        stan_init, stan_data = CmdStanPyBackend.prepare_data(res, data)
        assert stan_init['delta'] == [0.1, 0.1, 0.1, 0.1]
        assert stan_init['beta'] == [1.0, 2.0]
        assert stan_data['T'] == 3
