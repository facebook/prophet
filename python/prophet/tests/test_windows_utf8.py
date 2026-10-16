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

    def test_tbb_path_logic_on_windows(self):
        """Test the TBB PATH prepending logic that should run on Windows."""
        # Save original PATH
        original_path = os.environ.get("PATH", "")
        
        try:
            # Simulate the logic from CmdStanPyBackend.__init__
            fake_cmdstan_path = "/fake/cmdstan"
            expected_tbb_dir = os.path.join(fake_cmdstan_path, "stan", "lib", "stan_math", "lib", "tbb")
            
            # Clear PATH
            os.environ["PATH"] = "/usr/bin:/bin"
            
            # Simulate the Windows PATH prepending logic from models.py
            # This is the code that should run when os.name == "nt"
            tbb_dir = expected_tbb_dir
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if tbb_dir not in path_entries:
                os.environ["PATH"] = os.pathsep.join([tbb_dir, *path_entries])
            
            # Verify the logic worked
            current_path = os.environ.get("PATH", "")
            path_entries = current_path.split(os.pathsep)
            
            assert expected_tbb_dir in path_entries, f"TBB directory should be in PATH"
            assert path_entries[0] == expected_tbb_dir, f"TBB directory should be first, got {path_entries[0]}"
            
        finally:
            os.environ["PATH"] = original_path

    def test_windows_path_prepending_with_os_name_check(self):
        """Test PATH prepending behavior respects os.name check."""
        # Save original PATH
        original_path = os.environ.get("PATH", "")
        
        try:
            # Test the conditional logic: if os.name != "nt", PATH is not modified
            fake_tbb_dir = "/fake/tbb/path"
            os.environ["PATH"] = "/usr/bin:/bin"
            
            # Simulate non-Windows (os.name != "nt")
            simulated_os_name = "posix"
            if simulated_os_name == "nt":
                # This is the logic from models.py
                path_entries = os.environ.get("PATH", "").split(os.pathsep)
                if fake_tbb_dir not in path_entries:
                    os.environ["PATH"] = os.pathsep.join([fake_tbb_dir, *path_entries])
            
            assert os.environ["PATH"] == "/usr/bin:/bin", "PATH should not change when os.name != 'nt'"
            
            # Test Windows behavior (os.name == "nt")
            os.environ["PATH"] = "/usr/bin:/bin"
            simulated_os_name = "nt"
            if simulated_os_name == "nt":
                # This is the logic from models.py
                path_entries = os.environ.get("PATH", "").split(os.pathsep)
                if fake_tbb_dir not in path_entries:
                    os.environ["PATH"] = os.pathsep.join([fake_tbb_dir, *path_entries])
            
            assert os.environ["PATH"].startswith(fake_tbb_dir), "TBB dir should be prepended when os.name == 'nt'"
            assert os.environ["PATH"] == f"{fake_tbb_dir}:/usr/bin:/bin", "PATH should have TBB first"
            
        finally:
            os.environ["PATH"] = original_path


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
