import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark tests as slow (include in run with --test-slow)")


def pytest_addoption(parser):
    parser.addoption("--test-slow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-slow"):
        return
    skip_slow = pytest.mark.skip(reason="Skipped due to the lack of '--test-slow' argument")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
