# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
import pandas as pd

from fbprophet.diagnostics import performance_metrics


logging.basicConfig()
logger = logging.getLogger(__name__)

try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import MonthLocator, num2date
    from matplotlib.ticker import FuncFormatter
except ImportError:
    logger.error('Importing matplotlib failed. Plotting will not work.')


def plot(
    m, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y',
):
    """Plot the Prophet forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Optional boolean to plot uncertainty intervals.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis

    Returns
    -------
    A matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = fcst['ds'].dt.to_pydatetime()
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.')
    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')
    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty:
        ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                        color='#0072B2', alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_components(
    m, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0,
):
    """Plot the Prophet forecast components.

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.

    Returns
    -------
    A matplotlib figure.
    """
    # Identify components to be plotted
    components = ['trend']
    if m.holidays is not None and 'holidays' in fcst:
        components.append('holidays')
    components.extend([name for name in m.seasonalities
                    if name in fcst])
    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components.append('extra_regressors_{}'.format(mode))
    npanel = len(components)

    fig, axes = plt.subplots(npanel, 1, facecolor='w',
                            figsize=(9, 3 * npanel))

    if npanel == 1:
        axes = [axes]

    multiplicative_axes = []

    for ax, plot_name in zip(axes, components):
        if plot_name == 'trend':
            plot_forecast_component(
                m=m, fcst=fcst, name='trend', ax=ax, uncertainty=uncertainty,
                plot_cap=plot_cap,
            )
        elif plot_name == 'weekly':
            plot_weekly(
                m=m, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start,
            )
        elif plot_name == 'yearly':
            plot_yearly(
                m=m, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start,
            )
        elif plot_name in [
            'holidays',
            'extra_regressors_additive',
            'extra_regressors_multiplicative',
        ]:
            plot_forecast_component(
                m=m, fcst=fcst, name=plot_name, ax=ax, uncertainty=uncertainty,
                plot_cap=False,
            )
        else:
            plot_seasonality(
                m=m, name=plot_name, ax=ax, uncertainty=uncertainty,
            )
        if plot_name in m.component_modes['multiplicative']:
            multiplicative_axes.append(ax)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(
    m, fcst, name, ax=None, uncertainty=True, plot_cap=False,
):
    """Plot a particular component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    ax: Optional matplotlib Axes to plot on.
    uncertainty: Optional boolean to plot uncertainty intervals.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    artists += ax.plot(fcst_t, fcst[name], ls='-', c='#0072B2')
    if 'cap' in fcst and plot_cap:
        artists += ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty:
        artists += [ax.fill_between(
            fcst_t, fcst[name + '_lower'], fcst[name + '_upper'],
            color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    ax.set_ylabel(name)
    if name in m.component_modes['multiplicative']:
        ax = set_y_as_percent(ax)
    return artists


def seasonality_plot_df(m, ds):
    """Prepare dataframe for plotting seasonal components.

    Parameters
    ----------
    m: Prophet model.
    ds: List of dates for column ds.

    Returns
    -------
    A dataframe with seasonal components on ds.
    """
    df_dict = {'ds': ds, 'cap': 1., 'floor': 0.}
    for name in m.extra_regressors:
        df_dict[name] = 0.
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df


def plot_weekly(m, ax=None, uncertainty=True, weekly_start=0):
    """Plot the weekly component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if this
        is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weekly_start))
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days = days.weekday_name
    artists += ax.plot(range(len(days)), seas['weekly'], ls='-',
                    c='#0072B2')
    if uncertainty:
        artists += [ax.fill_between(range(len(days)),
                                    seas['weekly_lower'], seas['weekly_upper'],
                                    color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day of week')
    ax.set_ylabel('weekly')
    if m.seasonalities['weekly']['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_yearly(m, ax=None, uncertainty=True, yearly_start=0):
    """Plot the yearly component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=365) +
            pd.Timedelta(days=yearly_start))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists += ax.plot(
        df_y['ds'].dt.to_pydatetime(), seas['yearly'], ls='-', c='#0072B2')
    if uncertainty:
        artists += [ax.fill_between(
            df_y['ds'].dt.to_pydatetime(), seas['yearly_lower'],
            seas['yearly_upper'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel('Day of year')
    ax.set_ylabel('yearly')
    if m.seasonalities['yearly']['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_seasonality(m, name, ax=None, uncertainty=True):
    """Plot a custom seasonal component.

    Parameters
    ----------
    m: Prophet model.
    name: Seasonality name, like 'daily', 'weekly'.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    plot_points = 200
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists += ax.plot(df_y['ds'].dt.to_pydatetime(), seas[name], ls='-',
                        c='#0072B2')
    if uncertainty:
        artists += [ax.fill_between(
            df_y['ds'].dt.to_pydatetime(), seas[name + '_lower'],
            seas[name + '_upper'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    xticks = pd.to_datetime(np.linspace(start.value, end.value, 7)
        ).to_pydatetime()
    ax.set_xticks(xticks)
    if period <= 2:
        fmt_str = '{dt:%T}'
    elif period < 14:
        fmt_str = '{dt:%m}/{dt:%d} {dt:%R}'
    else:
        fmt_str = '{dt:%m}/{dt:%d}'
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: fmt_str.format(dt=num2date(x))))
    ax.set_xlabel('ds')
    ax.set_ylabel(name)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def set_y_as_percent(ax):
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticklabels(yticklabels)
    return ax


def add_changepoints_to_plot(
    ax, m, fcst, threshold=0.01, cp_color='r', cp_linestyle='--', trend=True,
):
    """Add markers for significant changepoints to prophet forecast plot.
    
    Example:
    fig = m.plot(forecast)
    add_changepoints_to_plot(fig.gca(), m, forecast)
    
    Parameters
    ----------
    ax: axis on which to overlay changepoint markers.
    m: Prophet model.
    fcst: Forecast output from m.predict.
    threshold: Threshold on trend change magnitude for significance.
    cp_color: Color of changepoint markers.
    cp_linestyle: Linestyle for changepoint markers.
    trend: If True, will also overlay the trend.
    
    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if trend:
        artists.append(ax.plot(fcst['ds'], fcst['trend'], c=cp_color))
    signif_changepoints = m.changepoints[
        np.abs(np.nanmean(m.params['delta'], axis=0)) >= threshold
    ]
    for cp in signif_changepoints:
        artists.append(ax.axvline(x=cp, c=cp_color, ls=cp_linestyle))
    return artists


def plot_cross_validation_metric(df_cv, metric, rolling_window=0.1, ax=None):
    """Plot a performance metric vs. forecast horizon from cross validation.

    Cross validation produces a collection of out-of-sample model predictions
    that can be compared to actual values, at a range of different horizons
    (distance from the cutoff). This computes a specified performance metric
    for each prediction, and aggregated over a rolling window with horizon.

    This uses fbprophet.diagnostics.performance_metrics to compute the metrics.
    Valid values of metric are 'mse', 'rmse', 'mae', 'mape', and 'coverage'.

    rolling_window is the proportion of data included in the rolling window of
    aggregation. The default value of 0.1 means 10% of data are included in the
    aggregation for computing the metric.

    As a concrete example, if metric='mse', then this plot will show the
    squared error for each cross validation prediction, along with the MSE
    averaged over rolling windows of 10% of the data.

    Parameters
    ----------
    df_cv: The output from fbprophet.diagnostics.cross_validation.
    metric: Metric name, one of ['mse', 'rmse', 'mae', 'mape', 'coverage'].
    rolling_window: Proportion of data to use for rolling average of metric.
        In [0, 1]. Defaults to 0.1.
    ax: Optional matplotlib axis on which to plot. If not given, a new figure
        will be created.

    Returns
    -------
    a matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    # Get the metric at the level of individual predictions, and with the rolling window.
    df_none = performance_metrics(df_cv, metrics=[metric], rolling_window=0)
    df_h = performance_metrics(df_cv, metrics=[metric], rolling_window=rolling_window)

    # Some work because matplotlib does not handle timedelta
    # Target ~10 ticks.
    tick_w = max(df_none['horizon'].astype('timedelta64[ns]')) / 10.
    # Find the largest time resolution that has <1 unit per bin.
    dts = ['D', 'h', 'm', 's', 'ms', 'us', 'ns']
    dt_names = [
        'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
        'nanoseconds'
    ]
    dt_conversions = [
        24 * 60 * 60 * 10 ** 9,
        60 * 60 * 10 ** 9,
        60 * 10 ** 9,
        10 ** 9,
        10 ** 6,
        10 ** 3,
        1.,
    ]
    for i, dt in enumerate(dts):
        if np.timedelta64(1, dt) < np.timedelta64(tick_w, 'ns'):
            break

    x_plt = df_none['horizon'].astype('timedelta64[ns]').astype(np.int64) / float(dt_conversions[i])
    x_plt_h = df_h['horizon'].astype('timedelta64[ns]').astype(np.int64) / float(dt_conversions[i])

    ax.plot(x_plt, df_none[metric], '.', alpha=0.5, c='gray')
    ax.plot(x_plt_h, df_h[metric], '-', c='b')
    ax.grid(True)

    ax.set_xlabel('Horizon ({})'.format(dt_names[i]))
    ax.set_ylabel(metric)
    return fig
