# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from tqdm.auto import tqdm
from copy import deepcopy
import concurrent.futures
import multiprocessing
import sys
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from prophet.metrics import (  # noqa: F401 – re-exported for backward compatibility
    PERFORMANCE_METRICS,
    coverage,
    mae,
    mape,
    mdape,
    mse,
    performance_metrics,
    register_performance_metric,
    rmse,
    rolling_mean_by_h,
    rolling_median_by_h,
    smape,
)

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        Iterable,
        Protocol,
        TypeVar,
        type_check_only,
    )

    import dask.distributed as dd

    from .forecaster import Prophet

    _ModelT = TypeVar('_ModelT', bound=Prophet)

    _PerformanceMetricT = TypeVar("_PerformanceMetricT", bound=Callable[..., Any])

    @type_check_only
    class _SupportsMap(Protocol):
        def map(self, __f: Callable[..., object], *its: Iterable[object]) -> Any: ...


logger: logging.Logger = logging.getLogger('prophet')


def generate_cutoffs(
    df: pd.DataFrame,
    horizon: pd.Timedelta,
    initial: pd.Timedelta,
    period: pd.Timedelta,
) -> list[pd.Timestamp]:
    """Generate cutoff dates

    Parameters
    ----------
    df: pd.DataFrame with historical data.
    horizon: pd.Timedelta forecast horizon.
    initial: pd.Timedelta window of the initial forecast period.
    period: pd.Timedelta simulated forecasts are done with this period.

    Returns
    -------
    list of pd.Timestamp
    """
    # Last cutoff is 'latest date in data - horizon' date
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]
    while result[-1] >= min(df['ds']) + initial:
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
            # Next cutoff point is 'last date before cutoff in data - horizon'
            if cutoff > df['ds'].min():
                closest_date = df[df['ds'] <= cutoff].max()['ds']
                cutoff = closest_date - horizon
            # else no data left, leave cutoff as is, it will be dropped.
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            'Less data than horizon after initial window. '
            'Make horizon or initial shorter.'
        )
    logger.info('Making {} forecasts with cutoffs between {} and {}'.format(
        len(result), result[-1], result[0]
    ))
    return list(reversed(result))


def cross_validation(
    model: Prophet,
    horizon: str | pd.Timedelta,
    period: str | pd.Timedelta | None = None,
    initial: str | pd.Timedelta | None = None,
    parallel: Literal["processes", "threads", "dask"] | _SupportsMap | None = None,
    cutoffs: list[pd.Timestamp] | None = None,
    disable_tqdm: bool = False,
    extra_output_columns: str | list[str] | None = None,
) -> pd.DataFrame:
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points, which user can input.
    If not provided, begins from (end - horizon) and works backwards, making
    cutoffs with a spacing of period until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted Prophet model.
    horizon: string with pd.Timedelta compatible style, e.g., '5 days',
        '3 hours', '10 seconds'.
    period: string with pd.Timedelta compatible style. Simulated forecast will
        be done at every this period. If not provided, 0.5 * horizon is used.
    initial: string with pd.Timedelta compatible style. The first training
        period will include at least this much data. If not provided,
        3 * horizon is used.
    cutoffs: list of pd.Timestamp specifying cutoffs to be used during
        cross validation. If not provided, they are generated as described
        above.
    parallel: {None, 'processes', 'threads', 'dask', object}
        How to parallelize the forecast computation. By default no parallelism
        is used.

        * None: No parallelism.
        * 'processes': Parallelize with concurrent.futures.ProcessPoolExecutor.
        * 'threads': Parallelize with concurrent.futures.ThreadPoolExecutor.
            Note that some operations currently hold Python's Global Interpreter
            Lock, so parallelizing with threads may be slower than training
            sequentially.
        * 'dask': Parallelize with Dask.
           This requires that a dask.distributed Client be created.
        * object: Any instance with a `.map` method. This method will
          be called with :func:`single_cutoff_forecast` and a sequence of
          iterables where each element is the tuple of arguments to pass to
          :func:`single_cutoff_forecast`

          .. code-block::

             class MyBackend:
                 def map(self, func, *iterables):
                     results = [
                        func(*args)
                        for args in zip(*iterables)
                     ]
                     return results

    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None
    extra_output_columns: A String or List of Strings e.g. 'trend' or ['trend'].
         Additional columns to 'yhat' and 'ds' to be returned in output.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """

    if model.history is None:
        raise Exception('Model has not been fit. Fitting the model provides contextual parameters for cross validation.')

    df = model.history.copy().reset_index(drop=True)
    horizon = pd.Timedelta(horizon)
    predict_columns = ['ds', 'yhat']

    if model.uncertainty_samples:
        predict_columns.extend(['yhat_lower', 'yhat_upper'])

    if extra_output_columns is not None:
        if isinstance(extra_output_columns, str):
            extra_output_columns = [extra_output_columns]
        predict_columns.extend([c for c in extra_output_columns if c not in predict_columns])

    # Identify the largest seasonality period
    period_max = 0.
    for s in model.seasonalities.values():
        period_max = max(period_max, s['period'])
    seasonality_dt = pd.Timedelta(str(period_max) + ' days')

    if cutoffs is None:
        # Set period
        period = 0.5 * horizon if period is None else pd.Timedelta(period)

        # Set initial
        initial = (
            max(3 * horizon, seasonality_dt) if initial is None
            else pd.Timedelta(initial)
        )

        # Compute Cutoffs
        cutoffs = generate_cutoffs(df, horizon, initial, period)
    else:
        # add validation of the cutoff to make sure that the min cutoff is strictly greater than the min date in the history
        if min(cutoffs) <= df['ds'].min():
            raise ValueError("Minimum cutoff value is not strictly greater than min date in history")
        # max value of cutoffs is <= (end date minus horizon)
        end_date_minus_horizon = df['ds'].max() - horizon
        if max(cutoffs) > end_date_minus_horizon:
            raise ValueError("Maximum cutoff value is greater than end date minus horizon, no value for cross-validation remaining")
        initial = cutoffs[0] - df['ds'].min()

    # Check if the initial window
    # (that is, the amount of time between the start of the history and the first cutoff)
    # is less than the maximum seasonality period
    if initial < seasonality_dt:
            msg = 'Seasonality has period of {} days '.format(period_max)
            msg += 'which is larger than initial window. '
            msg += 'Consider increasing initial.'
            logger.warning(msg)

    if parallel:
        valid = {"threads", "processes", "dask"}

        if parallel == "threads":
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == "processes":
            if sys.platform.startswith("win") or sys.platform == "darwin":
                ctx = multiprocessing.get_context("spawn")
            else:
                ctx = multiprocessing.get_context("forkserver")
            pool = concurrent.futures.ProcessPoolExecutor(mp_context=ctx)
        elif parallel == "dask":
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requires the optional "
                                  "dependency dask.") from e
            pool = get_client()
            # get_client() returns a client managed by the caller.
            # delay df and model to avoid large objects in task graph.
            df, model = pool.scatter([df, model])
        elif hasattr(parallel, "map"):
            pool = parallel
        else:
            msg = ("'parallel' should be one of {} for an instance with a "
                   "'map' method".format(', '.join(valid)))
            raise ValueError(msg)

        iterables = ((df, model, cutoff, horizon, predict_columns)
                     for cutoff in cutoffs)
        iterables = zip(*iterables)

        logger.info("Applying in parallel with %s", pool)
        try:
            predicts = pool.map(single_cutoff_forecast, *iterables)
            if parallel == "dask":
                # convert Futures to DataFrames
                predicts = cast("dd.Client", pool).gather(predicts)
        finally:
            if (
                parallel in ("threads", "processes")
                and isinstance(pool, concurrent.futures.Executor)
            ):
                pool.shutdown(wait=True)

    else:
        predicts = [
            single_cutoff_forecast(df, model, cutoff, horizon, predict_columns)
            for cutoff in (tqdm(cutoffs) if not disable_tqdm else cutoffs)
        ]

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return pd.concat(cast("Iterable[Any]", predicts), axis=0).reset_index(drop=True)


def single_cutoff_forecast(
    df: pd.DataFrame,
    model: Prophet,
    cutoff: pd.Timestamp,
    horizon: pd.Timedelta,
    predict_columns: list[str],
) -> pd.DataFrame:
    """Forecast for single cutoff. Used in cross validation function
    when evaluating for multiple cutoffs either sequentially or in parallel.

    Parameters
    ----------
    df: pd.DataFrame.
        DataFrame with history to be used for single
        cutoff forecast.
    model: Prophet model object.
    cutoff: pd.Timestamp cutoff date.
        Simulated Forecast will start from this date.
    horizon: pd.Timedelta forecast horizon.
    predict_columns: List of strings e.g. ['ds', 'yhat'].
        Columns with date and forecast to be returned in output.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.

    """

    # Generate new object with copying fitting options
    m = prophet_copy(model, cutoff)
    # Train model
    history_c = df[df['ds'] <= cutoff]
    if history_c.shape[0] < 2:
        raise Exception(
            'Less than two datapoints before cutoff. '
            'Increase initial window.'
        )
    m.fit(history_c, **model.fit_kwargs)
    # Calculate yhat
    index_predicted = (df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)
    # Get the columns for the future dataframe
    columns = ['ds']
    if m.growth == 'logistic':
        columns.append('cap')
        if m.logistic_floor:
            columns.append('floor')
    columns.extend(m.extra_regressors.keys())
    columns.extend([
        props['condition_name']
        for props in m.seasonalities.values()
        if props['condition_name'] is not None])
    yhat = m.predict(df[index_predicted][columns])
    # Merge yhat(predicts), y(df, original data) and cutoff

    assert m.stan_backend
    m.stan_backend.cleanup()

    return pd.concat([
        yhat[predict_columns],
        df[index_predicted][['y']].reset_index(drop=True),
        pd.DataFrame({'cutoff': [cutoff] * len(yhat)})
    ], axis=1)


def prophet_copy(m: _ModelT, cutoff: pd.Timestamp | None = None) -> _ModelT:
    """Copy Prophet object

    Parameters
    ----------
    m: Prophet model.
    cutoff: pd.Timestamp or None, default None.
        cutoff Timestamp for changepoints member variable.
        changepoints are only retained if 'changepoints <= cutoff'

    Returns
    -------
    Prophet class object with the same parameter with model variable
    """
    if m.history is None:
        raise Exception('This is for copying a fitted Prophet object.')

    if m.specified_changepoints:
        changepoints = m.changepoints
        if cutoff is not None:
            # Filter change points '< cutoff'
            last_history_date = max(m.history['ds'][m.history['ds'] <= cutoff])
            assert changepoints is not None
            changepoints = changepoints[changepoints < last_history_date]
    else:
        changepoints = None

    # Auto seasonalities are set to False because they are already set in
    # m.seasonalities.
    m2 = m.__class__(
        growth=m.growth,
        n_changepoints=m.n_changepoints,
        changepoint_range=m.changepoint_range,
        changepoints=changepoints,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=m.holidays,
        holidays_mode=m.holidays_mode,
        seasonality_mode=m.seasonality_mode,
        seasonality_prior_scale=m.seasonality_prior_scale,
        changepoint_prior_scale=m.changepoint_prior_scale,
        holidays_prior_scale=m.holidays_prior_scale,
        mcmc_samples=m.mcmc_samples,
        interval_width=m.interval_width,
        uncertainty_samples=m.uncertainty_samples,
        stan_backend=(
            m.stan_backend.get_type() if m.stan_backend is not None
            else None
        ),
    )
    m2.extra_regressors = deepcopy(m.extra_regressors)
    m2.seasonalities = deepcopy(m.seasonalities)
    m2.country_holidays = deepcopy(m.country_holidays)
    return m2



