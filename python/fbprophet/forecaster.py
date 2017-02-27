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
import pickle

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd

# fb-block 1 start
import pkg_resources
# fb-block 1 end

try:
    import pystan
except ImportError:
    print('You cannot run prophet without pystan installed')
    raise

# fb-block 2

class Prophet(object):
    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            yearly_seasonality=True,
            weekly_seasonality=True,
            holidays=None,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
    ):
        if growth not in ('linear', 'logistic'):
            raise ValueError("growth setting must be 'linear' or 'logistic'")

        self.growth = growth

        self.changepoints = pd.to_datetime(changepoints)
        if self.changepoints is not None:
            self.n_changepoints = len(self.changepoints)
        else:
            self.n_changepoints = n_changepoints

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality

        if holidays is not None:
            if not (
                isinstance(holidays, pd.DataFrame)
                and 'ds' in holidays
                and 'holiday' in holidays
            ):
                raise ValueError("holidays must be a DataFrame with 'ds' and "
                                 "'holiday' columns.")
            holidays['ds'] = pd.to_datetime(holidays['ds'])
        self.holidays = holidays

        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)

        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        # Set during fitting
        self.start = None
        self.end = None
        self.y_scale = None
        self.stan_fit = None
        self.params = {}
        self.history = None

    @classmethod
    def get_linear_model(cls):
        # fb-block 3
        # fb-block 4 start
        model_file = pkg_resources.resource_filename(
            'fbprophet',
            'stan_models/linear_growth.pkl'
        )
        # fb-block 4 end
        with open(model_file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def get_logistic_model(cls):
        # fb-block 5
        # fb-block 6 start
        model_file = pkg_resources.resource_filename(
            'fbprophet',
            'stan_models/logistic_growth.pkl'
        )
        # fb-block 6 end
        with open(model_file, 'rb') as f:
            return pickle.load(f)

    def setup_dataframe(self, df, initialize_scales=False):
        """Create auxillary columns 't', 't_ix', 'y_scaled', and 'cap_scaled'.

        These columns are used during both fitting and prediction.
        """
        if 'y' in df:
            df['y'] = pd.to_numeric(df['y'])
        df['ds'] = pd.to_datetime(df['ds'])

        df = df.sort_values('ds')

        if initialize_scales:
            self.y_scale = df['y'].max()
            self.start, self.end = df['ds'].min(), df['ds'].max()

        t_scale = self.end - self.start

        df['t'] = (df['ds'] - self.start) / t_scale
        if 'y' in df:
            df['y_scaled'] = df['y'] / self.y_scale

        if self.growth == 'logistic':
            assert 'cap' in df
            df['cap_scaled'] = df['cap'] / self.y_scale

        return df

    def set_changepoints(self):
        """Generate a list of changepoints.

        Either:
        1) the changepoints were passed in explicitly
           A) they are empty
           B) not empty, needs validation
        2) we are generating a grid of them
        3) the user prefers no changepoints to be used
        """
        if self.changepoints is not None:
            if len(self.changepoints) == 0:
                pass
            else:
                too_low = min(self.changepoints) < self.history['ds'].min()
                too_high = max(self.changepoints) > self.history['ds'].max()
                if too_low or too_high:
                    raise ValueError('Changepoints must fall within training data.')
        elif self.n_changepoints > 0:
            # Place potential changepoints evenly throuh first 80% of history
            max_ix = np.floor(self.history.shape[0] * 0.8)
            cp_indexes = (
                np.linspace(0, max_ix, self.n_changepoints + 1)
                .round()
                .astype(np.int)
            )
            self.changepoints = self.history.ix[cp_indexes]['ds'].tail(-1)
        else:
            # set empty changepoints
            self.changepoints = []

    def get_changepoint_indexes(self):
        if len(self.changepoints) == 0:
            return np.array([0])  # a dummy changepoint
        else:
            row_index = pd.DatetimeIndex(self.history['ds'])
            indexes = []
            for cp in self.changepoints:
                # In the future this may raise a KeyError, but for now we
                # should guarantee that all changepoint dates are included in
                # the historical data.
                indexes.append(row_index.get_loc(cp))
            return np.array(indexes).astype(np.int)

    def get_changepoint_times(self):
        cpi = self.get_changepoint_indexes()
        return np.array(self.history['t'].iloc[cpi])

    def get_changepoint_matrix(self):
        changepoint_indexes = self.get_changepoint_indexes()

        A = np.zeros((self.history.shape[0], len(changepoint_indexes)))
        for i, index in enumerate(changepoint_indexes):
            A[index:self.history.shape[0], i] = 1

        return A

    @staticmethod
    def fourier_series(dates, period, series_order):
        """Generate a Fourier expansion for a fixed frequency and order.

        Parameters
        ----------
        dates: a pd.Series containing timestamps
        period: an integer frequency (number of days)
        series_order: number of components to generate

        Returns
        -------
        a 2-dimensional np.array with one row per row in `dt`
        """
        # convert to days since epoch
        t = np.array(
            (dates - pd.datetime(1970, 1, 1))
            .apply(lambda x: x.days)
            .astype(np.float)
        )
        return np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(series_order)
            for fun in (np.sin, np.cos)
        ])

    @classmethod
    def make_seasonality_features(cls, dates, period, series_order, prefix):
        features = cls.fourier_series(dates, period, series_order)
        columns = [
            '{}_{}'.format(prefix, i + 1)
            for i in range(features.shape[1])
        ]
        return pd.DataFrame(features, columns=columns)

    def make_holiday_features(self, dates):
        """Generate a DataFrame with each column corresponding to a holiday.
        """
        # A smaller prior scale will shrink holiday estimates more
        scale_ratio = self.holidays_prior_scale / self.seasonality_prior_scale
        # Holds columns of our future matrix.
        expanded_holidays = defaultdict(lambda: np.zeros(dates.shape[0]))
        # Makes an index so we can perform `get_loc` below.
        row_index = pd.DatetimeIndex(dates)

        for ix, row in self.holidays.iterrows():
            dt = row.ds.date()
            try:
                lw = int(row.get('lower_window', 0))
                uw = int(row.get('upper_window', 0))
            except ValueError:
                lw = 0
                uw = 0
            for offset in range(lw, uw + 1):
                occurrence = dt + timedelta(days=offset)
                try:
                    loc = row_index.get_loc(occurrence)
                except KeyError:
                    loc = None

                key = '{}_{}{}'.format(
                    row.holiday,
                    '+' if offset >= 0 else '-',
                    abs(offset)
                )
                if loc is not None:
                    expanded_holidays[key][loc] = scale_ratio
                else:
                    # Access key to generate value
                    expanded_holidays[key]

        # This relies pretty importantly on pandas keeping the columns in order.
        return pd.DataFrame(expanded_holidays)


    def make_all_seasonality_features(self, df):
        seasonal_features = [
            # Add a column of zeros in case no seasonality is used.
            pd.DataFrame({'zeros': np.zeros(df.shape[0])})
        ]

        # Seasonality features
        if self.yearly_seasonality:
            seasonal_features.append(self.make_seasonality_features(
                df['ds'],
                365.25,
                10,
                'yearly',
            ))

        if self.weekly_seasonality:
            seasonal_features.append(self.make_seasonality_features(
                df['ds'],
                7,
                3,
                'weekly',
            ))

        if self.holidays is not None:
            seasonal_features.append(self.make_holiday_features(df['ds']))
        return pd.concat(seasonal_features, axis=1)

    @staticmethod
    def linear_growth_init(df):
        i0, i1 = df['ds'].idxmin(), df['ds'].idxmax()
        T = df['t'].ix[i1] - df['t'].ix[i0]
        k = (df['y_scaled'].ix[i1] - df['y_scaled'].ix[i0]) / T
        m = df['y_scaled'].ix[i0] - k * df['t'].ix[i0]
        return (k, m)

    @staticmethod
    def logistic_growth_init(df):
        i0, i1 = df['ds'].idxmin(), df['ds'].idxmax()
        T = df['t'].ix[i1] - df['t'].ix[i0]

        # Force valid values, in case y > cap.
        r0 = max(1.01, df['cap_scaled'].ix[i0] / df['y_scaled'].ix[i0])
        r1 = max(1.01, df['cap_scaled'].ix[i1] / df['y_scaled'].ix[i1])

        if abs(r0 - r1) <= 0.01:
            r0 = 1.05 * r0

        L0 = np.log(r0 - 1)
        L1 = np.log(r1 - 1)

        # Initialize the offset
        m = L0 * T / (L0 - L1)
        # And the rate
        k = L0 / m
        return (k, m)

    # fb-block 7
    def fit(self, df, **kwargs):
        """Fit the Prophet model to data.

        Parameters
        ----------
        df: pd.DataFrame containing history. Must have columns 'ds', 'y', and
            if logistic growth, 'cap'.
        kwargs: Additional arguments passed to Stan's sampling or optimizing
            function, as appropriate.

        Returns
        -------
        The fitted Prophet object.
        """
        history = df[df['y'].notnull()].copy()
        history.reset_index(inplace=True, drop=True)

        history = self.setup_dataframe(history, initialize_scales=True)
        self.history = history
        seasonal_features = self.make_all_seasonality_features(history)

        self.set_changepoints()
        A = self.get_changepoint_matrix()
        changepoint_indexes = self.get_changepoint_indexes()

        dat = {
            'T': history.shape[0],
            'K': seasonal_features.shape[1],
            'S': len(changepoint_indexes),
            'y': history['y_scaled'],
            't': history['t'],
            'A': A,
            # Need to add one because Stan is 1-indexed.
            's_indx': changepoint_indexes + 1,
            'X': seasonal_features,
            'sigma': self.seasonality_prior_scale,
            'tau': self.changepoint_prior_scale,
        }

        if self.growth == 'linear':
            kinit = self.linear_growth_init(history)
            model = self.get_linear_model()
        else:
            dat['cap'] = history['cap_scaled']
            kinit = self.logistic_growth_init(history)
            model = self.get_logistic_model()

        def stan_init():
            return {
                'k': kinit[0],
                'm': kinit[1],
                'delta': np.zeros(len(changepoint_indexes)),
                'beta': np.zeros(seasonal_features.shape[1]),
                'sigma_obs': 1,
            }

        if self.mcmc_samples > 0:
            stan_fit = model.sampling(
                dat,
                init=stan_init,
                iter=self.mcmc_samples,
                **kwargs
            )
            for par in stan_fit.model_pars:
                self.params[par] = stan_fit[par]

        else:
            params = model.optimizing(dat, init=stan_init, iter=1e4, **kwargs)
            for par in params:
                self.params[par] = params[par].reshape((1, -1))

        # If no changepoints were requested, replace delta with 0s
        if len(self.changepoints) == 0:
            # Fold delta into the base rate k
            params['k'] = params['k'] + params['delta']
            params['delta'] = np.zeros(params['delta'].shape)

        return self

    # fb-block 8
    def predict(self, df=None):
        """Predict historical and future values for y.

        Note: you must only pass in future dates here.
        Historical dates are prepended before predictions are made.

        `df` can be None, in which case we predict only on history.
        """
        if df is None:
            df = self.history
        else:
            df = self.setup_dataframe(df)

        df['trend'] = self.predict_trend(df)
        seasonal_components = self.predict_seasonal_components(df)
        intervals = self.predict_uncertainty(df)

        df2 = pd.concat((df, intervals, seasonal_components), axis=1)
        df2['yhat'] = df2['trend'] + df2['seasonal']
        return df2

    @staticmethod
    def piecewise_linear(t, deltas, k, m, changepoint_ts):
        # Intercept changes
        gammas = -changepoint_ts * deltas
        # Get cumulative slope and intercept at each t
        k_t = k * np.ones_like(t)
        m_t = m * np.ones_like(t)
        for s, t_s in enumerate(changepoint_ts):
            indx = t >= t_s
            k_t[indx] += deltas[s]
            m_t[indx] += gammas[s]
        return k_t * t + m_t

    @staticmethod
    def piecewise_logistic(t, cap, deltas, k, m, changepoint_ts):
        # Compute offset changes
        k_cum = np.concatenate((np.atleast_1d(k), np.cumsum(deltas) + k))
        gammas = np.zeros(len(changepoint_ts))
        for i, t_s in enumerate(changepoint_ts):
            gammas[i] = (
                (t_s - m - np.sum(gammas))
                * (1 - k_cum[i] / k_cum[i + 1])
            )
        # Get cumulative rate and offset at each t
        k_t = k * np.ones_like(t)
        m_t = m * np.ones_like(t)
        for s, t_s in enumerate(changepoint_ts):
            indx = t >= t_s
            k_t[indx] += deltas[s]
            m_t[indx] += gammas[s]
        return cap / (1 + np.exp(-k_t * (t - m_t)))

    def predict_trend(self, df):
        k = np.nanmean(self.params['k'])
        m = np.nanmean(self.params['m'])
        deltas = np.nanmean(self.params['delta'], axis=0)

        t = np.array(df['t'])
        cpts = self.get_changepoint_times()
        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, cpts)
        else:
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(t, cap, deltas, k, m, cpts)

        return trend * self.y_scale

    def predict_seasonal_components(self, df):
        seasonal_features = self.make_all_seasonality_features(df)
        lower_p = 100 * (1.0 - self.interval_width) / 2
        upper_p = 100 * (1.0 + self.interval_width) / 2

        components = pd.DataFrame({
            'col': np.arange(seasonal_features.shape[1]),
            'component': [x.split('_')[0] for x in seasonal_features.columns],
        })
        # Remove the placeholder
        components = components[components['component'] != 'zeros']

        if components.shape[0] > 0:
            X = seasonal_features.as_matrix()
            data = {}
            for component, features in components.groupby('component'):
                cols = features.col.tolist()
                comp_beta = self.params['beta'][:, cols]
                comp_features = X[:, cols]
                comp = (
                    np.matmul(comp_features, comp_beta.transpose())
                    * self.y_scale
                )
                data[component] = np.nanmean(comp, axis=1)
                data[component + '_lower'] = np.nanpercentile(comp, lower_p,
                                                              axis=1)
                data[component + '_upper'] = np.nanpercentile(comp, upper_p,
                                                              axis=1)

            component_predictions = pd.DataFrame(data)
            component_predictions['seasonal'] = (
                component_predictions[components['component'].unique()].sum(1))
        else:
            component_predictions = pd.DataFrame(
                {'seasonal': np.zeros(df.shape[0])})
        return component_predictions

    def predict_uncertainty(self, df):
        n_iterations = self.params['k'].shape[0]
        samp_per_iter = max(1, int(np.ceil(
            self.uncertainty_samples / float(n_iterations)
        )))

        # Generate seasonality features once so we can re-use them.
        seasonal_features = self.make_all_seasonality_features(df)

        sim_values = {'yhat': [], 'trend': [], 'seasonal': []}
        for i in range(n_iterations):
            for j in range(samp_per_iter):
                sim = self.sample_model(df, seasonal_features, i)
                for key in sim_values:
                    sim_values[key].append(sim[key])

        lower_p = 100 * (1.0 - self.interval_width) / 2
        upper_p = 100 * (1.0 + self.interval_width) / 2

        series = {}
        for key, value in sim_values.items():
            mat = np.column_stack(value)
            series['{}_lower'.format(key)] = np.nanpercentile(mat, lower_p,
                                                              axis=1)
            series['{}_upper'.format(key)] = np.nanpercentile(mat, upper_p,
                                                              axis=1)

        return pd.DataFrame(series)

    def sample_model(self, df, seasonal_features, iteration):
        trend = self.sample_predictive_trend(df, iteration)

        beta = self.params['beta'][iteration]
        seasonal = np.matmul(seasonal_features.as_matrix(), beta) * self.y_scale

        sigma = self.params['sigma_obs'][iteration]
        noise = np.random.normal(0, sigma, df.shape[0]) * self.y_scale

        return pd.DataFrame({
            'yhat': trend + seasonal + noise,
            'trend': trend,
            'seasonal': seasonal,
        })

    def sample_predictive_trend(self, df, iteration):
        k = self.params['k'][iteration]
        m = self.params['m'][iteration]
        deltas = self.params['delta'][iteration]

        t = np.array(df['t'])
        changepoint_ts = self.get_changepoint_times()
        T = t.max()

        if T > 1:
            # Get the time discretization of the history
            dt = np.diff(self.history['t'])
            dt = np.min(dt[dt > 0])
            # Number of time periods in the future
            N = np.ceil((T - 1) / float(dt))
            S = len(changepoint_ts)

            prob_change = min(1, (S * (T - 1)) / N)
            n_changes = np.random.binomial(N, prob_change)

            # Sample ts
            changepoint_ts_new = sorted(np.random.uniform(1, T, n_changes))
        else:
            # Case where we're not extrapolating.
            changepoint_ts_new = []
            n_changes = 0

        # Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
        lambda_ = np.mean(np.abs(deltas)) + 1e-8

        # Sample deltas
        deltas_new = np.random.laplace(0, lambda_, n_changes)

        # Prepend the times and deltas from the history
        changepoint_ts = np.concatenate((changepoint_ts, changepoint_ts_new))
        deltas = np.concatenate((deltas, deltas_new))

        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, changepoint_ts)
        else:
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(t, cap, deltas, k, m,
                                            changepoint_ts)

        return trend * self.y_scale

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        last_date = self.history['ds'].max()
        dates = pd.date_range(
            start=last_date,
            periods=periods + 1,  # closed='right' removes a period
            freq=freq,
            closed='right')  # omits the start date

        if include_history:
            dates = np.concatenate((np.array(self.history['ds']), dates))

        return pd.DataFrame({'ds': dates})

    def plot(self, fcst, uncertainty=True, xlabel='ds', ylabel='y'):
        """Plot the Prophet forecast.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        uncertainty: Optional boolean to plot uncertainty intervals.
        xlabel: Optional label name on X-axis
        ylabel: Optional label name on Y-axis

        Returns
        -------
        a matplotlib figure.
        """
        forecast_color = '#0072B2'
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(self.history['ds'].values, self.history['y'], 'k.')
        ax.plot(fcst['ds'].values, fcst['yhat'], ls='-', c=forecast_color)
        if 'cap' in fcst:
            ax.plot(fcst['ds'].values, fcst['cap'], ls='--', c='k')
        if uncertainty:
            ax.fill_between(fcst['ds'].values, fcst['yhat_lower'],
                            fcst['yhat_upper'], color=forecast_color, alpha=0.2)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig

    def plot_components(self, fcst, uncertainty=True):
        """Plot the Prophet forecast components.

        Will plot whichever are available of: trend, holidays, weekly
        seasonality, and yearly seasonality.

        Parameters
        ----------
        fcst: pd.DataFrame output of self.predict.
        uncertainty: Optional boolean to plot uncertainty intervals.

        Returns
        -------
        a matplotlib figure.
        """
        # Identify components to be plotted
        plot_trend = True
        plot_holidays = self.holidays is not None
        plot_weekly = 'weekly' in fcst
        plot_yearly = 'yearly' in fcst

        npanel = plot_trend + plot_holidays + plot_weekly + plot_yearly
        forecast_color = '#0072B2'
        fig = plt.figure(facecolor='w', figsize=(9, 3 * npanel))
        panel_num = 1
        ax = fig.add_subplot(npanel, 1, panel_num)
        ax.plot(fcst['ds'].values, fcst['trend'], ls='-', c=forecast_color)
        if 'cap' in fcst:
            ax.plot(fcst['ds'].values, fcst['cap'], ls='--', c='k')
        if uncertainty:
            ax.fill_between(
                fcst['ds'].values, fcst['trend_lower'], fcst['trend_upper'],
                color=forecast_color, alpha=0.2)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.set_xlabel('ds')
        ax.set_ylabel('trend')

        if plot_holidays:
            panel_num += 1
            ax = fig.add_subplot(npanel, 1, panel_num)
            holiday_comps = self.holidays['holiday'].unique()
            y_holiday = fcst[holiday_comps].sum(1)
            y_holiday_l = fcst[[h + '_lower' for h in holiday_comps]].sum(1)
            y_holiday_u = fcst[[h + '_upper' for h in holiday_comps]].sum(1)
            # NOTE the above CI calculation is incorrect if holidays overlap
            # in time. Since it is just for the visualization we will not
            # worry about it now.
            ax.plot(fcst['ds'].values, y_holiday, ls='-', c=forecast_color)
            if uncertainty:
                ax.fill_between(fcst['ds'].values, y_holiday_l, y_holiday_u,
                                color=forecast_color, alpha=0.2)
            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
            ax.set_xlabel('ds')
            ax.set_ylabel('holidays')

        if plot_weekly:
            panel_num += 1
            ax = fig.add_subplot(npanel, 1, panel_num)
            df_s = fcst.copy()
            df_s['dow'] = df_s['ds'].dt.weekday_name
            df_s = df_s.groupby('dow').first()
            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday']
            y_weekly = [df_s.loc[d]['weekly'] for d in days]
            y_weekly_l = [df_s.loc[d]['weekly_lower'] for d in days]
            y_weekly_u = [df_s.loc[d]['weekly_upper'] for d in days]
            ax.plot(range(len(days)), y_weekly, ls='-', c=forecast_color)
            if uncertainty:
                ax.fill_between(range(len(days)), y_weekly_l, y_weekly_u,
                                color=forecast_color, alpha=0.2)
            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            ax.set_xticks(range(len(days)))
            ax.set_xticklabels(days)
            ax.set_xlabel('Day of week')
            ax.set_ylabel('weekly')

        if plot_yearly:
            panel_num += 1
            ax = fig.add_subplot(npanel, 1, panel_num)
            df_s = fcst.copy()
            df_s['doy'] = df_s['ds'].map(lambda x: x.strftime('2000-%m-%d'))
            df_s = df_s.groupby('doy').first().sort_index()
            ax.plot(pd.to_datetime(df_s.index), df_s['yearly'], ls='-',
                    c=forecast_color)
            if uncertainty:
                ax.fill_between(
                    pd.to_datetime(df_s.index), df_s['yearly_lower'],
                    df_s['yearly_upper'], color=forecast_color, alpha=0.2)
            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
            ax.xaxis.set_major_formatter(DateFormatter('%B %-d'))
            ax.xaxis.set_major_locator(months)
            ax.set_xlabel('Day of year')
            ax.set_ylabel('yearly')

        fig.tight_layout()
        return fig

# fb-block 9
