# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from prophet import Prophet
from prophet.models import CmdStanPyBackend


class TestWindowsTbbPath:
    """Test that bundled TBB directory is prepended to PATH on Windows."""

    def test_cmdstanpy_backend_prepends_tbb_on_windows(self):
        """Test that CmdStanPyBackend actually prepends TBB to PATH on Windows."""
        import sys
        
        # Save original state
        original_path = os.environ.get("PATH", "")
        
        # Import prophet.models to get a reference
        from prophet import models as prophet_models
        
        # Save original os.name from the os module that models.py uses
        original_models_os = getattr(prophet_models, 'os', None)
        
        try:
            # Set up test environment
            os.environ["PATH"] = "/usr/bin:/bin"
            # Expected TBB path based on how the mock constructs paths
            expected_tbb_dir = f"/fake/prophet/stan_model/cmdstan-{CmdStanPyBackend.CMDSTAN_VERSION}/stan/lib/stan_math/lib/tbb"
            
            # Create a mock os module with name='nt'
            mock_os_module = MagicMock()
            mock_os_module.name = 'nt'
            mock_os_module.environ = os.environ
            mock_os_module.pathsep = os.pathsep
            
            # Inject mock os into prophet.models if it exists there, or patch at import level
            if original_models_os is not None:
                prophet_models.os = mock_os_module
            
            # Mock importlib.resources.files to return a fake cmdstan path
            with patch('prophet.models.importlib.resources.files') as mock_files:
                class MockResource:
                    def __init__(self, path):
                        self.path = path
                    
                    def __truediv__(self, other):
                        return MockResource(self.path + "/" + str(other))
                    
                    def __str__(self):
                        return self.path
                    
                    def exists(self):
                        return True
                
                # Return the root, and let the code build the path
                mock_files.return_value = MockResource("/fake/prophet")
                
                # Mock cmdstanpy module at import time
                mock_cmdstanpy = MagicMock()
                mock_cmdstanpy.set_cmdstan_path = MagicMock()
                
                with patch.dict('sys.modules', {'cmdstanpy': mock_cmdstanpy}):
                    # Use a class to track path construction
                    class MockPath:
                        def __init__(self, p):
                            self.path = str(p)
                        
                        def __truediv__(self, other):
                            return MockPath(self.path + "/" + str(other))
                        
                        def __str__(self):
                            return self.path
                    
                    # Mock pathlib.Path to avoid WindowsPath issues
                    with patch('prophet.models.pathlib.Path', MockPath):
                        # Mock load_model to avoid actual model loading
                        with patch.object(CmdStanPyBackend, 'load_model', return_value=MagicMock()):
                            # This should trigger the TBB PATH prepending on Windows
                            backend = CmdStanPyBackend()
            
            # Verify TBB directory was prepended to PATH
            current_path = os.environ.get("PATH", "")
            path_entries = current_path.split(os.pathsep)
            
            assert expected_tbb_dir in path_entries, \
                f"TBB directory '{expected_tbb_dir}' should be in PATH. PATH={current_path}"
            assert path_entries[0] == expected_tbb_dir, \
                f"TBB directory should be first in PATH. First entry is '{path_entries[0]}', expected '{expected_tbb_dir}'"
            
        finally:
            # Restore original state
            os.environ["PATH"] = original_path
            if original_models_os is not None:
                prophet_models.os = original_models_os
            elif hasattr(prophet_models, 'os'):
                delattr(prophet_models, 'os')


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
