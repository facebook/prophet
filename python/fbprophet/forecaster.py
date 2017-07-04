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
# fb-block 1 end

try:
    import pystan
except ImportError:
    logger.error('You cannot run prophet without pystan installed')
    raise

# fb-block 2



class Prophet(object):
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
        self.y_scale = None
        self.t_scale = None
        self.changepoints_t = None
        self.stan_fit = None
        self.params = {}
        self.history = None
        self.history_dates = None
        self.validate_inputs()

    def validate_inputs(self):
        """Validates the inputs to Prophet."""
        if self.growth not in ('linear', 'logistic'):
            raise ValueError(
                "Parameter 'growth' should be 'linear' or 'logistic'.")
        if self.holidays is not None:
            has_lower = 'lower_window' in self.holidays
            has_upper = 'upper_window' in self.holidays
            if has_lower + has_upper == 1:
                raise ValueError('Holidays must have both lower_window and ' +
                                 'upper_window, or neither')
            if has_lower:
                if max(self.holidays['lower_window']) > 0:
                    raise ValueError('Holiday lower_window should be <= 0')
                if min(self.holidays['upper_window']) < 0:
                    raise ValueError('Holiday upper_window should be >= 0')
            for h in self.holidays['holiday'].unique():
                if '_delim_' in h:
                    raise ValueError('Holiday name cannot contain "_delim_"')
                if h in ['zeros', 'yearly', 'weekly', 'yhat', 'seasonal',
                         'trend']:
                    raise ValueError('Holiday name {} reserved.'.format(h))

    def setup_dataframe(self, df, initialize_scales=False):
        """Prepare dataframe for fitting or predicting.

        Adds a time index and scales y. Creates auxiliary columns 't', 't_ix',
        'y_scaled', and 'cap_scaled'. These columns are used during both
        fitting and predicting.

        Parameters
        ----------
        df: pd.DataFrame with columns ds, y, and cap if logistic growth.
        initialize_scales: Boolean set scaling factors in self from df.

        Returns
        -------
        pd.DataFrame prepared for fitting or predicting.
        """
        if 'y' in df:
            df['y'] = pd.to_numeric(df['y'])
        df['ds'] = pd.to_datetime(df['ds'])
        if df['ds'].isnull().any():
            raise ValueError('Found NaN in column ds.')

        df = df.sort_values('ds')
        df.reset_index(inplace=True, drop=True)

        if initialize_scales:
            self.y_scale = df['y'].abs().max()
            self.start = df['ds'].min()
            self.t_scale = df['ds'].max() - self.start

        df['t'] = (df['ds'] - self.start) / self.t_scale
        if 'y' in df:
            df['y_scaled'] = df['y'] / self.y_scale

        if self.growth == 'logistic':
            assert 'cap' in df
            df['cap_scaled'] = df['cap'] / self.y_scale

        return df

    def set_changepoints(self):
        """Set changepoints

        Sets m$changepoints to the dates of changepoints. Either:
        1) The changepoints were passed in explicitly.
            A) They are empty.
            B) They are not empty, and need validation.
        2) We are generating a grid of them.
        3) The user prefers no changepoints be used.
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
            # Place potential changepoints evenly through first 80% of history
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
        if len(self.changepoints) > 0:
            self.changepoints_t = np.sort(np.array(
                (self.changepoints - self.start) / self.t_scale))
        else:
            self.changepoints_t = np.array([0])  # dummy changepoint

    def get_changepoint_matrix(self):
        """Gets changepoint matrix for history dataframe."""
        A = np.zeros((self.history.shape[0], len(self.changepoints_t)))
        for i, t_i in enumerate(self.changepoints_t):
            A[self.history['t'].values >= t_i, i] = 1
        return A

    @staticmethod
    def fourier_series(dates, period, series_order):
        """Provides Fourier series components with the specified frequency
        and order.

        Parameters
        ----------
        dates: pd.Series containing timestamps.
        period: Number of days of the period.
        series_order: Number of components.

        Returns
        -------
        Matrix with seasonality features.
        """
        # convert to days since epoch
        t = np.array(
            (dates - pd.datetime(1970, 1, 1))
            .dt.days
            .astype(np.float)
        )
        return np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / period))
            for i in range(series_order)
            for fun in (np.sin, np.cos)
        ])

    @classmethod
    def make_seasonality_features(cls, dates, period, series_order, prefix):
        """Data frame with seasonality features.

        Parameters
        ----------
        cls: Prophet class.
        dates: pd.Series containing timestamps.
        period: Number of days of the period.
        series_order: Number of components.
        prefix: Column name prefix.

        Returns
        -------
        pd.DataFrame with seasonality features.
        """
        features = cls.fourier_series(dates, period, series_order)
        columns = [
            '{}_delim_{}'.format(prefix, i + 1)
            for i in range(features.shape[1])
        ]
        return pd.DataFrame(features, columns=columns)

    def make_holiday_features(self, dates):
        """Construct a dataframe of holiday features.

        Parameters
        ----------
        dates: pd.Series containing timestamps used for computing seasonality.

        Returns
        -------
        pd.DataFrame with a column for each holiday.
        """
        # A smaller prior scale will shrink holiday estimates more
        scale_ratio = self.holidays_prior_scale / self.seasonality_prior_scale
        # Holds columns of our future matrix.
        expanded_holidays = defaultdict(lambda: np.zeros(dates.shape[0]))
        # Makes an index so we can perform `get_loc` below.
        row_index = pd.DatetimeIndex(dates)

        for _ix, row in self.holidays.iterrows():
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

                key = '{}_delim_{}{}'.format(
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
        """Dataframe with seasonality features.

        Parameters
        ----------
        df: pd.DataFrame with dates for computing seasonality features.

        Returns
        -------
        pd.DataFrame with seasonality.
        """
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

    def set_auto_seasonalities(self):
        """Set seasonalities that were left on auto.

        Turns on yearly seasonality if there is >=2 years of history.
        Turns on weekly seasonality if there is >=2 weeks of history, and the
        spacing between dates in the history is <7 days.
        """
        first = self.history['ds'].min()
        last = self.history['ds'].max()
        if self.yearly_seasonality == 'auto':
            if last - first < pd.Timedelta(days=730):
                self.yearly_seasonality = False
                logger.info('Disabling yearly seasonality. Run prophet with '
                            'yearly_seasonality=True to override this.')
            else:
                self.yearly_seasonality = True
        if self.weekly_seasonality == 'auto':
            dt = self.history['ds'].diff()
            min_dt = dt.iloc[dt.nonzero()[0]].min()
            if ((last - first < pd.Timedelta(weeks=2)) or
                    (min_dt >= pd.Timedelta(weeks=1))):
                self.weekly_seasonality = False
                logger.info('Disabling weekly seasonality. Run prophet with '
                            'weekly_seasonality=True to override this.')
            else:
                self.weekly_seasonality = True

    @staticmethod
    def linear_growth_init(df):
        """Initialize linear growth.

        Provides a strong initialization for linear growth by calculating the
        growth and offset parameters that pass the function through the first
        and last points in the time series.

        Parameters
        ----------
        df: pd.DataFrame with columns ds (date), y_scaled (scaled time series),
            and t (scaled time).

        Returns
        -------
        A tuple (k, m) with the rate (k) and offset (m) of the linear growth
        function.
        """
        i0, i1 = df['ds'].idxmin(), df['ds'].idxmax()
        T = df['t'].ix[i1] - df['t'].ix[i0]
        k = (df['y_scaled'].ix[i1] - df['y_scaled'].ix[i0]) / T
        m = df['y_scaled'].ix[i0] - k * df['t'].ix[i0]
        return (k, m)

    @staticmethod
    def logistic_growth_init(df):
        """Initialize logistic growth.

        Provides a strong initialization for logistic growth by calculating the
        growth and offset parameters that pass the function through the first
        and last points in the time series.

        Parameters
        ----------
        df: pd.DataFrame with columns ds (date), cap_scaled (scaled capacity),
            y_scaled (scaled time series), and t (scaled time).

        Returns
        -------
        A tuple (k, m) with the rate (k) and offset (m) of the logistic growth
        function.
        """
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
        """Fit the Prophet model.

        This sets self.params to contain the fitted model parameters. It is a
        dictionary parameter names as keys and the following items:
            k (Mx1 array): M posterior samples of the initial slope.
            m (Mx1 array): The initial intercept.
            delta (MxN array): The slope change at each of N changepoints.
            beta (MxK matrix): Coefficients for K seasonality features.
            sigma_obs (Mx1 array): Noise level.
        Note that M=1 if MAP estimation.

        Parameters
        ----------
        df: pd.DataFrame containing the history. Must have columns ds (date
            type) and y, the time series. If self.growth is 'logistic', then
            df must also have a column cap that specifies the capacity at
            each ds.
        kwargs: Additional arguments passed to the optimizing or sampling
            functions in Stan.

        Returns
        -------
        The fitted Prophet object.
        """
        if self.history is not None:
            raise Exception('Prophet object can only be fit once. '
                            'Instantiate a new object.')
        history = df[df['y'].notnull()].copy()
        if np.isinf(history['y'].values).any():
            raise ValueError('Found infinity in column y.')
        self.history_dates = pd.to_datetime(df['ds']).sort_values()

        history = self.setup_dataframe(history, initialize_scales=True)
        self.history = history
        self.set_auto_seasonalities()
        seasonal_features = self.make_all_seasonality_features(history)

        self.set_changepoints()
        A = self.get_changepoint_matrix()

        dat = {
            'T': history.shape[0],
            'K': seasonal_features.shape[1],
            'S': len(self.changepoints_t),
            'y': history['y_scaled'],
            't': history['t'],
            'A': A,
            't_change': self.changepoints_t,
            'X': seasonal_features,
            'sigma': self.seasonality_prior_scale,
            'tau': self.changepoint_prior_scale,
        }

        if self.growth == 'linear':
            kinit = self.linear_growth_init(history)
        else:
            dat['cap'] = history['cap_scaled']
            kinit = self.logistic_growth_init(history)

        model = prophet_stan_models[self.growth]

        def stan_init():
            return {
                'k': kinit[0],
                'm': kinit[1],
                'delta': np.zeros(len(self.changepoints_t)),
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
            try:
                params = model.optimizing(
                    dat, init=stan_init, iter=1e4, **kwargs)
            except RuntimeError:
                params = model.optimizing(
                    dat, init=stan_init, iter=1e4, algorithm='Newton',
                    **kwargs
                )
            for par in params:
                self.params[par] = params[par].reshape((1, -1))

        # If no changepoints were requested, replace delta with 0s
        if len(self.changepoints) == 0:
            # Fold delta into the base rate k
            self.params['k'] = self.params['k'] + self.params['delta']
            self.params['delta'] = np.zeros(self.params['delta'].shape)

        return self

    # fb-block 8
    def predict(self, df=None):
        """Predict using the prophet model.

        Parameters
        ----------
        df: pd.DataFrame with dates for predictions (column ds), and capacity
            (column cap) if logistic growth. If not provided, predictions are
            made on the history.

        Returns
        -------
        A pd.DataFrame with the forecast components.
        """
        if df is None:
            df = self.history.copy()
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
        """Evaluate the piecewise linear function.

        Parameters
        ----------
        t: np.array of times on which the function is evaluated.
        deltas: np.array of rate changes at each changepoint.
        k: Float initial rate.
        m: Float initial offset.
        changepoint_ts: np.array of changepoint times.

        Returns
        -------
        Vector y(t).
        """
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
        """Evaluate the piecewise logistic function.

        Parameters
        ----------
        t: np.array of times on which the function is evaluated.
        cap: np.array of capacities at each t.
        deltas: np.array of rate changes at each changepoint.
        k: Float initial rate.
        m: Float initial offset.
        changepoint_ts: np.array of changepoint times.

        Returns
        -------
        Vector y(t).
        """
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
        """Predict trend using the prophet model.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Vector with trend on prediction dates.
        """
        k = np.nanmean(self.params['k'])
        m = np.nanmean(self.params['m'])
        deltas = np.nanmean(self.params['delta'], axis=0)

        t = np.array(df['t'])
        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, self.changepoints_t)
        else:
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(
                t, cap, deltas, k, m, self.changepoints_t)

        return trend * self.y_scale

    def predict_seasonal_components(self, df):
        """Predict seasonality broken down into components.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Dataframe with seasonal components.
        """
        seasonal_features = self.make_all_seasonality_features(df)
        lower_p = 100 * (1.0 - self.interval_width) / 2
        upper_p = 100 * (1.0 + self.interval_width) / 2

        components = pd.DataFrame({
            'col': np.arange(seasonal_features.shape[1]),
            'component': [x.split('_delim_')[0] for x in seasonal_features.columns],
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
        """Predict seasonality broken down into components.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Dataframe with uncertainty intervals.
        """
        n_iterations = self.params['k'].shape[0]
        samp_per_iter = max(1, int(np.ceil(
            self.uncertainty_samples / float(n_iterations)
        )))

        # Generate seasonality features once so we can re-use them.
        seasonal_features = self.make_all_seasonality_features(df)

        sim_values = {'yhat': [], 'trend': [], 'seasonal': []}
        for i in range(n_iterations):
            for _j in range(samp_per_iter):
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
        """Simulate observations from the extrapolated generative model.

        Parameters
        ----------
        df: Prediction dataframe.
        seasonal_features: pd.DataFrame of seasonal features.
        iteration: Int sampling iteration to use parameters from.

        Returns
        -------
        Dataframe with trend, seasonality, and yhat, each like df['t'].
        """
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
        """Simulate the trend using the extrapolated generative model.

        Parameters
        ----------
        df: Prediction dataframe.
        seasonal_features: pd.DataFrame of seasonal features.
        iteration: Int sampling iteration to use parameters from.

        Returns
        -------
        np.array of simulated trend over df['t'].
        """
        k = self.params['k'][iteration]
        m = self.params['m'][iteration]
        deltas = self.params['delta'][iteration]

        t = np.array(df['t'])
        T = t.max()

        if T > 1:
            # Get the time discretization of the history
            dt = np.diff(self.history['t'])
            dt = np.min(dt[dt > 0])
            # Number of time periods in the future
            N = np.ceil((T - 1) / float(dt))
            S = len(self.changepoints_t)

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
        changepoint_ts = np.concatenate((self.changepoints_t,
                                         changepoint_ts_new))
        deltas = np.concatenate((deltas, deltas_new))

        if self.growth == 'linear':
            trend = self.piecewise_linear(t, deltas, k, m, changepoint_ts)
        else:
            cap = df['cap_scaled']
            trend = self.piecewise_logistic(t, cap, deltas, k, m,
                                            changepoint_ts)

        return trend * self.y_scale

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        """Simulate the trend using the extrapolated generative model.

        Parameters
        ----------
        periods: Int number of periods to forecast forward.
        freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
        include_history: Boolean to include the historical dates in the data
            frame for predictions.

        Returns
        -------
        pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        last_date = self.history_dates.max()
        dates = pd.date_range(
            start=last_date,
            periods=periods + 1,  # An extra in case we include start
            freq=freq)
        dates = dates[dates > last_date]  # Drop start if equals last_date
        dates = dates[:periods]  # Return correct number of periods

        if include_history:
            dates = np.concatenate((np.array(self.history_dates), dates))

        return pd.DataFrame({'ds': dates})

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
