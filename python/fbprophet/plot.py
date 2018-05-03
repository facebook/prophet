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
    seasonality, and yearly seasonality.

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
    if len(m.extra_regressors) > 0 and 'extra_regressors' in fcst:
        components.append('extra_regressors')
    npanel = len(components)

    fig, axes = plt.subplots(npanel, 1, facecolor='w',
                            figsize=(9, 3 * npanel))

    if npanel == 1:
        axes = [axes]

    for ax, plot_name in zip(axes, components):
        if plot_name == 'trend':
            plot_forecast_component(
                m=m, fcst=fcst, name='trend', ax=ax, uncertainty=uncertainty,
                plot_cap=plot_cap,
            )
        elif plot_name == 'holidays':
            plot_forecast_component(
                m=m, fcst=fcst, name='holidays', ax=ax,
                uncertainty=uncertainty, plot_cap=False,
            )
        elif plot_name == 'weekly':
            plot_weekly(
                m=m, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start,
            )
        elif plot_name == 'yearly':
            plot_yearly(
                m=m, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start,
            )
        elif plot_name == 'extra_regressors':
            plot_forecast_component(
                m=m, fcst=fcst, name='extra_regressors', ax=ax,
                uncertainty=uncertainty, plot_cap=False,
            )
        else:
            plot_seasonality(
                m=m, name=plot_name, ax=ax, uncertainty=uncertainty,
            )

    fig.tight_layout()
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
    return artists


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
