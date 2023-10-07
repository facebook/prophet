# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pandas as pd

from prophet.diagnostics import performance_metrics

logger = logging.getLogger('prophet.plot')



try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
    )
    from matplotlib.ticker import FuncFormatter

    from pandas.plotting import deregister_matplotlib_converters
    deregister_matplotlib_converters()
except ImportError:
    logger.error('Importing matplotlib failed. Plotting will not work.')

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:
    logger.error('Importing plotly failed. Interactive plots will not work.')


def plot(
    m, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y',
    figsize=(10, 6), include_legend=False
):
    """Plot the Prophet forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    ax: Optional matplotlib axes on which to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis
    figsize: Optional tuple width, height in inches.
    include_legend: Optional boolean to add legend to the plot.

    Returns
    -------
    A matplotlib figure.
    """
    user_provided_ax = False if ax is None else True
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    fcst_t = fcst['ds'].dt.to_pydatetime()
    ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'], 'k.',
            label='Observed data points')
    ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2', label='Forecast')
    if 'cap' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['cap'], ls='--', c='k', label='Maximum capacity')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k', label='Minimum capacity')
    if uncertainty and m.uncertainty_samples:
        ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                        color='#0072B2', alpha=0.2, label='Uncertainty interval')
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if include_legend:
        ax.legend()
    if not user_provided_ax:
        fig.tight_layout()
    return fig


def plot_components(
    m, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0,
    figsize=None
):
    """Plot the Prophet forecast components.

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    A matplotlib figure.
    """
    # Identify components to be plotted
    components = ['trend']
    if m.train_holiday_names is not None and 'holidays' in fcst:
        components.append('holidays')
    # Plot weekly seasonality, if present
    if 'weekly' in m.seasonalities and 'weekly' in fcst:
        components.append('weekly')
    # Yearly if present
    if 'yearly' in m.seasonalities and 'yearly' in fcst:
        components.append('yearly')
    # Other seasonalities
    components.extend([
        name for name in sorted(m.seasonalities)
        if name in fcst and name not in ['weekly', 'yearly']
    ])
    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components.append('extra_regressors_{}'.format(mode))
    npanel = len(components)

    figsize = figsize if figsize else (9, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    multiplicative_axes = []

    dt = m.history['ds'].diff()
    min_dt = dt.iloc[dt.values.nonzero()[0]].min() 

    for ax, plot_name in zip(axes, components):
        if plot_name == 'trend':
            plot_forecast_component(
                m=m, fcst=fcst, name='trend', ax=ax, uncertainty=uncertainty,
                plot_cap=plot_cap,
            )
        elif plot_name in m.seasonalities:
            if (
                (plot_name == 'weekly' or m.seasonalities[plot_name]['period'] == 7)
                and (min_dt == pd.Timedelta(days=1))
            ):
                plot_weekly(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start
                )
            elif plot_name == 'yearly' or m.seasonalities[plot_name]['period'] == 365.25:
                plot_yearly(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start
                )
            else:
                plot_seasonality(
                    m=m, name=plot_name, ax=ax, uncertainty=uncertainty,
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
        if plot_name in m.component_modes['multiplicative']:
            multiplicative_axes.append(ax)

    fig.tight_layout()
    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)
    return fig


def plot_forecast_component(
    m, fcst, name, ax=None, uncertainty=True, plot_cap=False, figsize=(10, 6)
):
    """Plot a particular component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    ax: Optional matplotlib Axes to plot on.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    fcst_t = fcst['ds'].dt.to_pydatetime()
    artists += ax.plot(fcst_t, fcst[name], ls='-', c='#0072B2')
    if 'cap' in fcst and plot_cap:
        artists += ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(
            fcst_t, fcst[name + '_lower'], fcst[name + '_upper'],
            color='#0072B2', alpha=0.2)]
    # Specify formatting to workaround matplotlib issue #12925
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
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
    # Activate all conditional seasonality columns
    for props in m.seasonalities.values():
        if props['condition_name'] is not None:
            df_dict[props['condition_name']] = True
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df


def plot_weekly(m, ax=None, uncertainty=True, weekly_start=0, figsize=(10, 6), name='weekly'):
    """Plot the weekly component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if this
        is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    weekly_start: Optional int specifying the start day of the weekly
        seasonality plot. 0 (default) starts the week on Sunday. 1 shifts
        by 1 day to Monday, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if changed from default 'weekly'.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute weekly seasonality for a Sun-Sat sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weekly_start))
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days = days.day_name()
    artists += ax.plot(range(len(days)), seas[name], ls='-',
                    c='#0072B2')
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(range(len(days)),
                                    seas[name + '_lower'], seas[name + '_upper'],
                                    color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day of week')
    ax.set_ylabel(name)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_yearly(m, ax=None, uncertainty=True, yearly_start=0, figsize=(10, 6), name='yearly'):
    """Plot the yearly component of the forecast.

    Parameters
    ----------
    m: Prophet model.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    yearly_start: Optional int specifying the start day of the yearly
        seasonality plot. 0 (default) starts the year on Jan 1. 1 shifts
        by 1 day to Jan 2, and so on.
    figsize: Optional tuple width, height in inches.
    name: Name of seasonality component if previously changed from default 'yearly'.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    # Compute yearly seasonality for a Jan 1 - Dec 31 sequence of dates.
    days = (pd.date_range(start='2017-01-01', periods=365) +
            pd.Timedelta(days=yearly_start))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists += ax.plot(
        df_y['ds'].dt.to_pydatetime(), seas[name], ls='-', c='#0072B2')
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(
            df_y['ds'].dt.to_pydatetime(), seas[name + '_lower'],
            seas[name + '_upper'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel('Day of year')
    ax.set_ylabel(name)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def plot_seasonality(m, name, ax=None, uncertainty=True, figsize=(10, 6)):
    """Plot a custom seasonal component.

    Parameters
    ----------
    m: Prophet model.
    name: Seasonality name, like 'daily', 'weekly'.
    ax: Optional matplotlib Axes to plot on. One will be created if
        this is not provided.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    figsize: Optional tuple width, height in inches.

    Returns
    -------
    a list of matplotlib artists
    """
    artists = []
    if not ax:
        fig = plt.figure(facecolor='w', figsize=figsize)
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
    if uncertainty and m.uncertainty_samples:
        artists += [ax.fill_between(
            df_y['ds'].dt.to_pydatetime(), seas[name + '_lower'],
            seas[name + '_upper'], color='#0072B2', alpha=0.2)]
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    n_ticks = 8
    xticks = pd.to_datetime(np.linspace(start.value, end.value, n_ticks)
        ).to_pydatetime()
    ax.set_xticks(xticks)
    if name == 'yearly':
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x)))
        ax.set_xlabel('Day of year')
    elif name == 'weekly':
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%A}'.format(dt=num2date(x)))
        ax.set_xlabel('Day of Week')
    elif name == 'daily':
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%T}'.format(dt=num2date(x)))
        ax.set_xlabel('Hour of day')
    elif period <= 2:
        fmt = FuncFormatter(
            lambda x, pos=None: '{dt:%T}'.format(dt=num2date(x)))
        ax.set_xlabel('Hours')
    else:
        fmt = FuncFormatter(
            lambda x, pos=None: '{:.0f}'.format(pos * period / (n_ticks - 1)))
        ax.set_xlabel('Days')
    ax.xaxis.set_major_formatter(fmt)
    ax.set_ylabel(name)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        ax = set_y_as_percent(ax)
    return artists


def set_y_as_percent(ax):
    yticks = 100 * ax.get_yticks()
    yticklabels = ['{0:.4g}%'.format(y) for y in yticks]
    ax.set_yticks(ax.get_yticks().tolist())
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
    ] if len(m.changepoints) > 0 else []
    for cp in signif_changepoints:
        artists.append(ax.axvline(x=cp, c=cp_color, ls=cp_linestyle))
    return artists


def plot_cross_validation_metric(
    df_cv, metric, rolling_window=0.1, ax=None, figsize=(10, 6), color='b',
    point_color='gray'
):
    """Plot a performance metric vs. forecast horizon from cross validation.

    Cross validation produces a collection of out-of-sample model predictions
    that can be compared to actual values, at a range of different horizons
    (distance from the cutoff). This computes a specified performance metric
    for each prediction, and aggregated over a rolling window with horizon.

    This uses prophet.diagnostics.performance_metrics to compute the metrics.
    Valid values of metric are 'mse', 'rmse', 'mae', 'mape', and 'coverage'.

    rolling_window is the proportion of data included in the rolling window of
    aggregation. The default value of 0.1 means 10% of data are included in the
    aggregation for computing the metric.

    As a concrete example, if metric='mse', then this plot will show the
    squared error for each cross validation prediction, along with the MSE
    averaged over rolling windows of 10% of the data.

    Parameters
    ----------
    df_cv: The output from prophet.diagnostics.cross_validation.
    metric: Metric name, one of ['mse', 'rmse', 'mae', 'mape', 'coverage'].
    rolling_window: Proportion of data to use for rolling average of metric.
        In [0, 1]. Defaults to 0.1.
    ax: Optional matplotlib axis on which to plot. If not given, a new figure
        will be created.
    figsize: Optional tuple width, height in inches.
    color: Optional color for plot and error points, useful when plotting
        multiple model performances on one axis for comparison.

    Returns
    -------
    a matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    # Get the metric at the level of individual predictions, and with the rolling window.
    df_none = performance_metrics(df_cv, metrics=[metric], rolling_window=-1)
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

    x_plt = df_none['horizon'].astype('timedelta64[ns]').view(np.int64) / float(dt_conversions[i])
    x_plt_h = df_h['horizon'].astype('timedelta64[ns]').view(np.int64) / float(dt_conversions[i])

    ax.plot(x_plt, df_none[metric], '.', alpha=0.1, c=point_color)
    ax.plot(x_plt_h, df_h[metric], '-', c=color)
    ax.grid(True)

    ax.set_xlabel('Horizon ({})'.format(dt_names[i]))
    ax.set_ylabel(metric)
    return fig


def plot_plotly(m, fcst, uncertainty=True, plot_cap=True, trend=False, changepoints=False,
                changepoints_threshold=0.01, xlabel='ds', ylabel='y', figsize=(900, 600)):
    """Plot the Prophet forecast with Plotly offline.

    Plotting in Jupyter Notebook requires initializing plotly.offline.init_notebook_mode():
    >>> import plotly.offline as py
    >>> py.init_notebook_mode()
    Then the figure can be displayed using plotly.offline.iplot(...):
    >>> fig = plot_plotly(m, fcst)
    >>> py.iplot(fig)
    see https://plot.ly/python/offline/ for details

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    trend: Optional boolean to plot trend
    changepoints: Optional boolean to plot changepoints
    changepoints_threshold: Threshold on trend change magnitude for significance.
    xlabel: Optional label name on X-axis
    ylabel: Optional label name on Y-axis

    Returns
    -------
    A Plotly Figure.
    """
    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    actual_color = 'black'
    cap_color = 'black'
    trend_color = '#B23B00'
    line_width = 2
    marker_size = 4

    data = []
    # Add actual
    data.append(go.Scatter(
        name='Actual',
        x=m.history['ds'],
        y=m.history['y'],
        marker=dict(color=actual_color, size=marker_size),
        mode='markers'
    ))
    # Add lower bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip'
        ))
    # Add prediction
    data.append(go.Scatter(
        name='Predicted',
        x=fcst['ds'],
        y=fcst['yhat'],
        mode='lines',
        line=dict(color=prediction_color, width=line_width),
        fillcolor=error_color,
        fill='tonexty' if uncertainty and m.uncertainty_samples else 'none'
    ))
    # Add upper bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            fillcolor=error_color,
            fill='tonexty',
            hoverinfo='skip'
        ))
    # Add caps
    if 'cap' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    # Add trend
    if trend:
        data.append(go.Scatter(
            name='Trend',
            x=fcst['ds'],
            y=fcst['trend'],
            mode='lines',
            line=dict(color=trend_color, width=line_width),
        ))
    # Add changepoints
    if changepoints and len(m.changepoints) > 0:
        signif_changepoints = m.changepoints[
            np.abs(np.nanmean(m.params['delta'], axis=0)) >= changepoints_threshold
        ]
        data.append(go.Scatter(
            x=signif_changepoints,
            y=fcst.loc[fcst['ds'].isin(signif_changepoints), 'trend'],
            marker=dict(size=50, symbol='line-ns-open', color=trend_color,
                        line=dict(width=line_width)),
            mode='markers',
            hoverinfo='skip'
        ))

    layout = dict(
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
        yaxis=dict(
            title=ylabel
        ),
        xaxis=dict(
            title=xlabel,
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label='1w',
                         step='day',
                         stepmode='backward'),
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_components_plotly(
        m, fcst, uncertainty=True, plot_cap=True, figsize=(900, 200)):
    """Plot the Prophet forecast components using Plotly.
    See plot_plotly() for Plotly setup instructions

    Will plot whichever are available of: trend, holidays, weekly
    seasonality, yearly seasonality, and additive and multiplicative extra
    regressors.

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: Set the size for the subplots (in px).

    Returns
    -------
    A Plotly Figure.
    """

    # Identify components to plot and get their Plotly props
    components = {}
    components['trend'] = get_forecast_component_plotly_props(
        m, fcst, 'trend', uncertainty, plot_cap)
    if m.train_holiday_names is not None and 'holidays' in fcst:
        components['holidays'] = get_forecast_component_plotly_props(
            m, fcst, 'holidays', uncertainty)

    regressors = {'additive': False, 'multiplicative': False}
    for name, props in m.extra_regressors.items():
        regressors[props['mode']] = True
    for mode in ['additive', 'multiplicative']:
        if regressors[mode] and 'extra_regressors_{}'.format(mode) in fcst:
            components['extra_regressors_{}'.format(mode)] = get_forecast_component_plotly_props(
                m, fcst, 'extra_regressors_{}'.format(mode))
    for seasonality in m.seasonalities:
        components[seasonality] = get_seasonality_plotly_props(m, seasonality)

    # Create Plotly subplot figure and add the components to it
    fig = make_subplots(rows=len(components), cols=1, print_grid=False)
    fig['layout'].update(go.Layout(
        showlegend=False,
        width=figsize[0],
        height=figsize[1] * len(components)
    ))
    for i, name in enumerate(components):
        if i == 0:
            xaxis = fig['layout']['xaxis']
            yaxis = fig['layout']['yaxis']
        else:
            xaxis = fig['layout']['xaxis{}'.format(i + 1)]
            yaxis = fig['layout']['yaxis{}'.format(i + 1)]
        xaxis.update(components[name]['xaxis'])
        yaxis.update(components[name]['yaxis'])
        for trace in components[name]['traces']:
            fig.append_trace(trace, i + 1, 1)
    return fig


def plot_forecast_component_plotly(m, fcst, name, uncertainty=True, plot_cap=False, figsize=(900, 300)):
    """Plot an particular component of the forecast using Plotly.
    See plot_plotly() for Plotly setup instructions

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.
    figsize: The plot's size (in px).

    Returns
    -------
    A Plotly Figure.
    """
    props = get_forecast_component_plotly_props(m, fcst, name, uncertainty, plot_cap)
    layout = go.Layout(
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        xaxis=props['xaxis'],
        yaxis=props['yaxis']
    )
    fig = go.Figure(data=props['traces'], layout=layout)
    return fig


def plot_seasonality_plotly(m, name, uncertainty=True, figsize=(900, 300)):
    """Plot a custom seasonal component using Plotly.
    See plot_plotly() for Plotly setup instructions

    Parameters
    ----------
    m: Prophet model.
    name: Seasonality name, like 'daily', 'weekly'.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    figsize: Set the plot's size (in px).

    Returns
    -------
    A Plotly Figure.
    """
    props = get_seasonality_plotly_props(m, name, uncertainty)
    layout = go.Layout(
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        xaxis=props['xaxis'],
        yaxis=props['yaxis']
    )
    fig = go.Figure(data=props['traces'], layout=layout)
    return fig


def get_forecast_component_plotly_props(m, fcst, name, uncertainty=True, plot_cap=False):
    """Prepares a dictionary for plotting the selected forecast component with Plotly

    Parameters
    ----------
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.
    plot_cap: Optional boolean indicating if the capacity should be shown
        in the figure, if available.

    Returns
    -------
    A dictionary with Plotly traces, xaxis and yaxis
    """
    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    cap_color = 'black'
    zeroline_color = '#AAA'
    line_width = 2

    range_margin = (fcst['ds'].max() - fcst['ds'].min()) * 0.05
    range_x = [fcst['ds'].min() - range_margin, fcst['ds'].max() + range_margin]

    text = None
    mode = 'lines'
    if name == 'holidays':
        
        # Combine holidays into one hover text
        holidays = m.construct_holiday_dataframe(fcst['ds'])
        holiday_features, _, _ = m.make_holiday_features(fcst['ds'], holidays)
        holiday_features.columns = holiday_features.columns.str.replace('_delim_', '', regex=False)
        holiday_features.columns = holiday_features.columns.str.replace('+0', '', regex=False)
        text = pd.Series(data='', index=holiday_features.index)
        for holiday_feature, idxs in holiday_features.items():
            text[idxs.astype(bool) & (text != '')] += '<br>'  # Add newline if additional holiday
            text[idxs.astype(bool)] += holiday_feature

    traces = []
    traces.append(go.Scatter(
        name=name,
        x=fcst['ds'],
        y=fcst[name],
        mode=mode,
        line=go.scatter.Line(color=prediction_color, width=line_width),
        text=text,
    ))
    if uncertainty and m.uncertainty_samples and (fcst[name + '_upper'] != fcst[name + '_lower']).any():
        if mode == 'markers':
            traces[0].update(
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=fcst[name + '_upper'],
                    arrayminus=fcst[name + '_lower'],
                    width=0,
                    color=error_color
                )
            )
        else:
            traces.append(go.Scatter(
                name=name + '_upper',
                x=fcst['ds'],
                y=fcst[name + '_upper'],
                mode=mode,
                line=go.scatter.Line(width=0, color=error_color)
            ))
            traces.append(go.Scatter(
                name=name + '_lower',
                x=fcst['ds'],
                y=fcst[name + '_lower'],
                mode=mode,
                line=go.scatter.Line(width=0, color=error_color),
                fillcolor=error_color,
                fill='tonexty'
            ))
    if 'cap' in fcst and plot_cap:
        traces.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=go.scatter.Line(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        traces.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=go.scatter.Line(color=cap_color, dash='dash', width=line_width),
        ))

    xaxis = go.layout.XAxis(
        type='date',
        range=range_x)
    yaxis = go.layout.YAxis(rangemode='normal' if name == 'trend' else 'tozero',
                            title=go.layout.yaxis.Title(text=name),
                            zerolinecolor=zeroline_color)
    if name in m.component_modes['multiplicative']:
        yaxis.update(tickformat='%', hoverformat='.2%')
    return {'traces': traces, 'xaxis': xaxis, 'yaxis': yaxis}


def get_seasonality_plotly_props(m, name, uncertainty=True):
    """Prepares a dictionary for plotting the selected seasonality with Plotly

    Parameters
    ----------
    m: Prophet model.
    name: Name of the component to plot.
    uncertainty: Optional boolean to plot uncertainty intervals, which will
        only be done if m.uncertainty_samples > 0.

    Returns
    -------
    A dictionary with Plotly traces, xaxis and yaxis
    """
    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    line_width = 2
    zeroline_color = '#AAA'

    # Compute seasonality from Jan 1 through a single period.
    start = pd.to_datetime('2017-01-01 0000')
    period = m.seasonalities[name]['period']
    end = start + pd.Timedelta(days=period)
    if (m.history['ds'].dt.hour == 0).all():  # Day Precision
        plot_points = np.floor(period).astype(int)
    elif (m.history['ds'].dt.minute == 0).all():  # Hour Precision
        plot_points = np.floor(period * 24).astype(int)
    else:  # Minute Precision
        plot_points = np.floor(period * 24 * 60).astype(int)
    days = pd.to_datetime(np.linspace(start.value, end.value, plot_points, endpoint=False))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)

    traces = []
    traces.append(go.Scatter(
        name=name,
        x=df_y['ds'],
        y=seas[name],
        mode='lines',
        line=go.scatter.Line(color=prediction_color, width=line_width)
    ))
    if uncertainty and m.uncertainty_samples and (seas[name + '_upper'] != seas[name + '_lower']).any():
        traces.append(go.Scatter(
            name=name + '_upper',
            x=df_y['ds'],
            y=seas[name + '_upper'],
            mode='lines',
            line=go.scatter.Line(width=0, color=error_color)
        ))
        traces.append(go.Scatter(
            name=name + '_lower',
            x=df_y['ds'],
            y=seas[name + '_lower'],
            mode='lines',
            line=go.scatter.Line(width=0, color=error_color),
            fillcolor=error_color,
            fill='tonexty'
        ))

    # Set tick formats (examples are based on 2017-01-06 21:15)
    if period <= 2:
        tickformat = '%H:%M'  # "21:15"
    elif period < 7:
        tickformat = '%A %H:%M'  # "Friday 21:15"
    elif period < 14:
        tickformat = '%A'  # "Friday"
    else:
        tickformat = '%B %e'  # "January  6"

    range_margin = (df_y['ds'].max() - df_y['ds'].min()) * 0.05
    xaxis = go.layout.XAxis(
        tickformat=tickformat,
        type='date',
        range=[df_y['ds'].min() - range_margin, df_y['ds'].max() + range_margin]
    )

    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text=name),
                            zerolinecolor=zeroline_color)
    if m.seasonalities[name]['mode'] == 'multiplicative':
        yaxis.update(tickformat='%', hoverformat='.2%')

    return {'traces': traces, 'xaxis': xaxis, 'yaxis': yaxis}
