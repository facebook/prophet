# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import patch, MagicMock
import pytest

from prophet import Prophet
from prophet.models import CmdStanPyBackend


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
                assert path_entries[0] == expected_tbb_dir, f"TBB directory should be first in PATH"


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
