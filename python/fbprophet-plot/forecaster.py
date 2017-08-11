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

from collections import defaultdict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd

# fb-block 1 start
from fbprophet.models import prophet_stan_models
from fbprophet.forecaster import Prophet
# fb-block 1 end

try:
    import pystan
except ImportError:
    logger.error('You cannot run prophet without pystan installed')
    raise

# fb-block 2



class ProphetPlot(Prophet):
    """Prophet forecaster.

    Parameters
    ----------
    growth: String 'linear' or 'logistic' to specify a linear or logistic
        trend.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first 80 percent of the history.
    yearly_seasonality: Fit yearly seasonality. Can be 'auto', True, or False.
    weekly_seasonality: Fit weekly seasonality. Can be 'auto', True, or False.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays.
    seasonality_prior_scale: Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality.
    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model.
    changepoint_prior_scale: Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    interval_width: Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: Number of simulated draws used to estimate
        uncertainty intervals.
    """

    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            holidays=None,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
    ):
        Prophet.__init__(self, growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            holidays=holidays,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
    )

    @classmethod
    def fromProphet(cls, prophet):
        return cls(growth=prophet.growth,
            changepoints=prophet.changepoints,
            n_changepoints=prophet.n_changepoints
            yearly_seasonality=prophet.yearly_seasonality,
            weekly_seasonality=prophet.weekly_seasonality,
            holidays=prophet.holidays,
            seasonality_prior_scale=prophet.seasonality_prior_scale,
            holidays_prior_scale=prophet.holidays_prior_scale,
            changepoint_prior_scale=prophet.changepoint_prior_scale,
            mcmc_samples=prophet.mcmc_samples,
            interval_width=prophet.interval_width,
            uncertainty_samples=prophet.uncertainty_samples,
    )

    def plot(self, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds',
             ylabel='y'):
        """Plot the Prophet forecast.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
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
        ax.plot(self.history['ds'].values, self.history['y'], 'k.')
        ax.plot(fcst['ds'].values, fcst['yhat'], ls='-', c='#0072B2')
        if 'cap' in fcst and plot_cap:
            ax.plot(fcst['ds'].values, fcst['cap'], ls='--', c='k')
        if uncertainty:
            ax.fill_between(fcst['ds'].values, fcst['yhat_lower'],
                            fcst['yhat_upper'], color='#0072B2',
                            alpha=0.2)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig

    def plot_components(self, fcst, uncertainty=True, plot_cap=True,
                        weekly_start=0, yearly_start=0):
        """Plot the Prophet forecast components.

        Will plot whichever are available of: trend, holidays, weekly
        seasonality, and yearly seasonality.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
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
        components = [('trend', True),
                      ('holidays', self.holidays is not None),
                      ('weekly', 'weekly' in fcst),
                      ('yearly', 'yearly' in fcst)]
        components = [plot for plot, cond in components if cond]
        npanel = len(components)

        fig, axes = plt.subplots(npanel, 1, facecolor='w',
                                 figsize=(9, 3 * npanel))

        for ax, plot in zip(axes, components):
            if plot == 'trend':
                self.plot_trend(
                    fcst, ax=ax, uncertainty=uncertainty, plot_cap=plot_cap)
            elif plot == 'holidays':
                self.plot_holidays(fcst, ax=ax, uncertainty=uncertainty)
            elif plot == 'weekly':
                self.plot_weekly(
                    ax=ax, uncertainty=uncertainty, weekly_start=weekly_start)
            elif plot == 'yearly':
                self.plot_yearly(
                    ax=ax, uncertainty=uncertainty, yearly_start=yearly_start)

        fig.tight_layout()
        return fig

    def plot_trend(self, fcst, ax=None, uncertainty=True, plot_cap=True):
        """Plot the trend component of the forecast.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
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
        artists += ax.plot(fcst['ds'].values, fcst['trend'], ls='-',
                           c='#0072B2')
        if 'cap' in fcst and plot_cap:
            artists += ax.plot(fcst['ds'].values, fcst['cap'], ls='--', c='k')
        if uncertainty:
            artists += [ax.fill_between(
                fcst['ds'].values, fcst['trend_lower'], fcst['trend_upper'],
                color='#0072B2', alpha=0.2)]
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_xlabel('ds')
        ax.set_ylabel('trend')
        return artists

    def plot_holidays(self, fcst, ax=None, uncertainty=True):
        """Plot the holidays component of the forecast.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        ax: Optional matplotlib Axes to plot on. One will be created if this
            is not provided.
        uncertainty: Optional boolean to plot uncertainty intervals.

        Returns
        -------
        a list of matplotlib artists
        """
        artists = []
        if not ax:
            fig = plt.figure(facecolor='w', figsize=(10, 6))
            ax = fig.add_subplot(111)
        holiday_comps = self.holidays['holiday'].unique()
        y_holiday = fcst[holiday_comps].sum(1)
        y_holiday_l = fcst[[h + '_lower' for h in holiday_comps]].sum(1)
        y_holiday_u = fcst[[h + '_upper' for h in holiday_comps]].sum(1)
        # NOTE the above CI calculation is incorrect if holidays overlap
        # in time. Since it is just for the visualization we will not
        # worry about it now.
        artists += ax.plot(fcst['ds'].values, y_holiday, ls='-',
                           c='#0072B2')
        if uncertainty:
            artists += [ax.fill_between(fcst['ds'].values,
                                        y_holiday_l, y_holiday_u,
                                        color='#0072B2', alpha=0.2)]
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_xlabel('ds')
        ax.set_ylabel('holidays')
        return artists

    def plot_weekly(self, ax=None, uncertainty=True, weekly_start=0):
        """Plot the weekly component of the forecast.

        Parameters
        ----------
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
        df_w = pd.DataFrame({'ds': days, 'cap': 1.})
        df_w = self.setup_dataframe(df_w)
        seas = self.predict_seasonal_components(df_w)
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

    def plot_yearly(self, ax=None, uncertainty=True, yearly_start=0):
        """Plot the yearly component of the forecast.

        Parameters
        ----------
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
        df_y = pd.DataFrame(
            {'ds': pd.date_range(start='2017-01-01', periods=365) +
             pd.Timedelta(days=yearly_start), 'cap': 1.})
        df_y = self.setup_dataframe(df_y)
        seas = self.predict_seasonal_components(df_y)
        artists += ax.plot(df_y['ds'], seas['yearly'], ls='-',
                           c='#0072B2')
        if uncertainty:
            artists += [ax.fill_between(
                df_y['ds'].values, seas['yearly_lower'],
                seas['yearly_upper'], color='#0072B2', alpha=0.2)]
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
        ax.xaxis.set_major_locator(months)
        ax.set_xlabel('Day of year')
        ax.set_ylabel('yearly')
        return artists
