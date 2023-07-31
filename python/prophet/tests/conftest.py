from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="package")
def daily_univariate_ts() -> pd.DataFrame:
    """Daily univariate time series with 2 years of data"""
    return pd.read_csv(Path(__file__).parent / "data.csv", parse_dates=["ds"])


@pytest.fixture(scope="package")
def subdaily_univariate_ts() -> pd.DataFrame:
    """Sub-daily univariate time series"""
    return pd.read_csv(Path(__file__).parent / "data2.csv", parse_dates=["ds"])


@pytest.fixture(scope="package")
def large_numbers_ts() -> pd.DataFrame:
    """Univariate time series with large values to test scaling"""
    return pd.read_csv(Path(__file__).parent / "data3.csv", parse_dates=["ds"])


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark tests as slow (include in run with --test-slow)")


def pytest_addoption(parser):
    parser.addoption("--test-slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption(
        "--backend",
        nargs="+",
        default=["CMDSTANPY"],
        help="Probabilistic Programming Language backend to perform tests with.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-slow"):
        return
    skip_slow = pytest.mark.skip(reason="Skipped due to the lack of '--test-slow' argument")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_generate_tests(metafunc):
    """
    For each test, if `backend` is used as a fixture, add a parametrization equal to the value of the
    --backend option.

    This is used to re-run the test suite for different probabilistic programming language backends
    (e.g. cmdstanpy, numpyro).
    """
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", metafunc.config.getoption("backend"))
