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
import warnings
import numpy as np
import pandas as pd

from fbprophet.diagnostics import prophet_copy
from fbprophet.models import prophet_stan_model
from fbprophet.make_holidays import get_holiday_names, make_holidays_df
from fbprophet.plot import (
    plot,
    plot_components,
    plot_forecast_component,
    seasonality_plot_df,
    plot_weekly,
    plot_yearly,
    plot_seasonality,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
warnings.filterwarnings("default", category=DeprecationWarning)

try:
    import pystan  # noqa F401
except ImportError:
    logger.exception('You cannot run fbprophet without pystan installed')


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
        the first `changepoint_range` proportion of the history.
    changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    Not used if input `changepoints` is supplied.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    append_holidays: country name or abbreviation; must be string
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_prior_scale: Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
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
            changepoint_range=0.8,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            append_holidays=None,
            seasonality_mode='additive',
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
            self.specified_changepoints = True
        else:
            self.n_changepoints = n_changepoints
            self.specified_changepoints = False

        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality

        if holidays is not None:
            if not (
                isinstance(holidays, pd.DataFrame)
                and 'ds' in holidays  # noqa W503
                and 'holiday' in holidays  # noqa W503
            ):
                raise ValueError("holidays must be a DataFrame with 'ds' and "
                                 "'holiday' columns.")
            holidays['ds'] = pd.to_datetime(holidays['ds'])
        self.holidays = holidays

        if append_holidays is not None:
            if not (
                    isinstance(append_holidays, str)
            ):
                raise ValueError("append_holidays must be a string")
        self.append_holidays = append_holidays

        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)

        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        # Set during fitting
        self.start = None
        self.y_scale = None
        self.logistic_floor = False
        self.t_scale = None
        self.changepoints_t = None
        self.seasonalities = {}
        self.extra_regressors = {}
        self.stan_fit = None
        self.params = {}
        self.history = None
        self.history_dates = None
        self.train_component_cols = None
        self.component_modes = None
        self.train_holiday_names = None
        self.validate_inputs()

    def validate_inputs(self):
        """Validates the inputs to Prophet."""
        if self.growth not in ('linear', 'logistic'):
            raise ValueError(
                "Parameter 'growth' should be 'linear' or 'logistic'.")
        if ((self.changepoint_range < 0) or (self.changepoint_range > 1)):
            raise ValueError("Parameter 'changepoint_range' must be in [0, 1]")
        if self.holidays is not None:
            has_lower = 'lower_window' in self.holidays
            has_upper = 'upper_window' in self.holidays
            if has_lower + has_upper == 1:
                raise ValueError('Holidays must have both lower_window and ' +
                                 'upper_window, or neither')
            if has_lower:
                if self.holidays['lower_window'].max() > 0:
                    raise ValueError('Holiday lower_window should be <= 0')
                if self.holidays['upper_window'].min() < 0:
                    raise ValueError('Holiday upper_window should be >= 0')
            for h in self.holidays['holiday'].unique():
                self.validate_column_name(h, check_holidays=False)
        if self.seasonality_mode not in ['additive', 'multiplicative']:
            raise ValueError(
                "seasonality_mode must be 'additive' or 'multiplicative'"
            )

    def validate_column_name(self, name, check_holidays=True,
                             check_seasonalities=True, check_regressors=True):
        """Validates the name of a seasonality, holiday, or regressor.

        Parameters
        ----------
        name: string
        check_holidays: bool check if name already used for holiday
        check_seasonalities: bool check if name already used for seasonality
        check_regressors: bool check if name already used for regressor
        """
        if '_delim_' in name:
            raise ValueError('Name cannot contain "_delim_"')
        reserved_names = [
            'trend', 'additive_terms', 'daily', 'weekly', 'yearly',
            'holidays', 'zeros', 'extra_regressors_additive', 'yhat',
            'extra_regressors_multiplicative', 'multiplicative_terms',
        ]
        rn_l = [n + '_lower' for n in reserved_names]
        rn_u = [n + '_upper' for n in reserved_names]
        reserved_names.extend(rn_l)
        reserved_names.extend(rn_u)
        reserved_names.extend([
            'ds', 'y', 'cap', 'floor', 'y_scaled', 'cap_scaled'])
        if name in reserved_names:
            raise ValueError('Name "{}" is reserved.'.format(name))
        if (check_holidays and self.holidays is not None and
                name in self.holidays['holiday'].unique()):
            raise ValueError(
                'Name "{}" already used for a holiday.'.format(name))
        if (check_holidays and self.append_holidays is not None and
                name in get_holiday_names(self.append_holidays)):
            raise ValueError(
                'Name "{}" is a holiday name in {}.'.format(name, self.append_holidays))
        if check_seasonalities and name in self.seasonalities:
            raise ValueError(
                'Name "{}" already used for a seasonality.'.format(name))
        if check_regressors and name in self.extra_regressors:
            raise ValueError(
                'Name "{}" already used for an added regressor.'.format(name))

    def setup_dataframe(self, df, initialize_scales=False):
        """Prepare dataframe for fitting or predicting.

        Adds a time index and scales y. Creates auxiliary columns 't', 't_ix',
        'y_scaled', and 'cap_scaled'. These columns are used during both
        fitting and predicting.

        Parameters
        ----------
        df: pd.DataFrame with columns ds, y, and cap if logistic growth. Any
            specified additional regressors must also be present.
        initialize_scales: Boolean set scaling factors in self from df.

        Returns
        -------
        pd.DataFrame prepared for fitting or predicting.
        """
        if 'y' in df:
            df['y'] = pd.to_numeric(df['y'])
            if np.isinf(df['y'].values).any():
                raise ValueError('Found infinity in column y.')
        df['ds'] = pd.to_datetime(df['ds'])
        if df['ds'].isnull().any():
            raise ValueError('Found NaN in column ds.')
        for name in self.extra_regressors:
            if name not in df:
                raise ValueError(
                    'Regressor "{}" missing from dataframe'.format(name))

        df = df.sort_values('ds')
        df.reset_index(inplace=True, drop=True)

        self.initialize_scales(initialize_scales, df)

        if self.logistic_floor:
            if 'floor' not in df:
                raise ValueError("Expected column 'floor'.")
        else:
            df['floor'] = 0
        if self.growth == 'logistic':
            if 'cap' not in df:
                raise ValueError(
                    "Capacities must be supplied for logistic growth in "
                    "column 'cap'"
                )
            df['cap_scaled'] = (df['cap'] - df['floor']) / self.y_scale

        df['t'] = (df['ds'] - self.start) / self.t_scale
        if 'y' in df:
            df['y_scaled'] = (df['y'] - df['floor']) / self.y_scale

        for name, props in self.extra_regressors.items():
            df[name] = pd.to_numeric(df[name])
            df[name] = ((df[name] - props['mu']) / props['std'])
            if df[name].isnull().any():
                raise ValueError('Found NaN in column ' + name)
        return df

    def initialize_scales(self, initialize_scales, df):
        """Initialize model scales.

        Sets model scaling factors using df.

        Parameters
        ----------
        initialize_scales: Boolean set the scales or not.
        df: pd.DataFrame for setting scales.
        """
        if not initialize_scales:
            return
        if self.growth == 'logistic' and 'floor' in df:
            self.logistic_floor = True
            floor = df['floor']
        else:
            floor = 0.
        self.y_scale = (df['y'] - floor).abs().max()
        if self.y_scale == 0:
            self.y_scale = 1
        self.start = df['ds'].min()
        self.t_scale = df['ds'].max() - self.start
        for name, props in self.extra_regressors.items():
            standardize = props['standardize']
            n_vals = len(df[name].unique())
            if n_vals < 2:
                standardize = False
            if standardize == 'auto':
                if set(df[name].unique()) == set([1, 0]):
                    # Don't standardize binary variables.
                    standardize = False
                else:
                    standardize = True
            if standardize:
                mu = df[name].mean()
                std = df[name].std()
                self.extra_regressors[name]['mu'] = mu
                self.extra_regressors[name]['std'] = std

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
                    raise ValueError(
                        'Changepoints must fall within training data.')
        else:
            # Place potential changepoints evenly through first
            # changepoint_range proportion of the history
            hist_size = np.floor(
                self.history.shape[0] * self.changepoint_range)
            if self.n_changepoints + 1 > hist_size:
                self.n_changepoints = hist_size - 1
                logger.info(
                    'n_changepoints greater than number of observations.'
                    'Using {}.'.format(self.n_changepoints)
                )
            if self.n_changepoints > 0:
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                        .round()
                        .astype(np.int)
                )
                self.changepoints = (
                    self.history.iloc[cp_indexes]['ds'].tail(-1)
                )
            else:
                # set empty changepoints
                self.changepoints = []
        if len(self.changepoints) > 0:
            self.changepoints_t = np.sort(np.array(
                (self.changepoints - self.start) / self.t_scale))
        else:
            self.changepoints_t = np.array([0])  # dummy changepoint

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
                .dt.total_seconds()
                .astype(np.float)
        ) / (3600 * 24.)
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
        holiday_features: pd.DataFrame with a column for each holiday.
        prior_scale_list: List of prior scales for each holiday column.
        holiday_names: List of names of holidays
        """
        # Concatenate holidays and append_holidays
        all_holidays = self.holidays
        if self.append_holidays is not None:
            year_list = list({x.year for x in dates})
            append_holidays_df = make_holidays_df(
                                    year_list=year_list,
                                    country=self.append_holidays)
            all_holidays = pd.concat((all_holidays, append_holidays_df), sort=False)
            all_holidays.reset_index(drop=True, inplace=True)
        # Make fit and predict holidays components match
        if self.train_holiday_names is not None:
            train_holidays = self.train_holiday_names
            # Remove holiday names didn't show up in fit
            index_to_drop = all_holidays.index[
                                np.logical_not(
                                    all_holidays.holiday.isin(train_holidays))]
            all_holidays = all_holidays.drop(index_to_drop)
            # Add holiday names show up in fit but not in predict with ds as NA
            holidays_to_add = pd.DataFrame(
                                {'holiday':
                                    train_holidays[
                                        np.logical_not(
                                            train_holidays.isin(all_holidays.holiday))]})
            all_holidays = pd.concat((all_holidays, holidays_to_add), sort=False)
            all_holidays.reset_index(drop=True, inplace=True)

        # Holds columns of our future matrix.
        expanded_holidays = defaultdict(lambda: np.zeros(dates.shape[0]))
        prior_scales = {}
        # Makes an index so we can perform `get_loc` below.
        # Strip to just dates.
        row_index = pd.DatetimeIndex(dates.apply(lambda x: x.date()))

        for _ix, row in all_holidays.iterrows():
            dt = row.ds.date()
            try:
                lw = int(row.get('lower_window', 0))
                uw = int(row.get('upper_window', 0))
            except ValueError:
                lw = 0
                uw = 0
            ps = float(row.get('prior_scale', self.holidays_prior_scale))
            if np.isnan(ps):
                ps = float(self.holidays_prior_scale)
            if (
                    row.holiday in prior_scales and prior_scales[row.holiday] != ps
            ):
                raise ValueError(
                    'Holiday {} does not have consistent prior scale '
                    'specification.'.format(row.holiday))
            if ps <= 0:
                raise ValueError('Prior scale must be > 0')
            prior_scales[row.holiday] = ps

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
                    expanded_holidays[key][loc] = 1.
                else:
                    # Access key to generate value
                    expanded_holidays[key]
        holiday_features = pd.DataFrame(expanded_holidays)
        # Make sure fit and predict component_cols perfectly equal
        holiday_features = holiday_features[sorted(holiday_features.columns.tolist())]
        prior_scale_list = [
            prior_scales[h.split('_delim_')[0]]
            for h in holiday_features.columns
        ]
        holiday_names = list(prior_scales.keys())
        # Store holiday names used in fit
        if self.train_holiday_names is None:
            self.train_holiday_names = pd.Series(holiday_names)
        return holiday_features, prior_scale_list, holiday_names

    def add_regressor(
            self, name, prior_scale=None, standardize='auto', mode=None
    ):
        """Add an additional regressor to be used for fitting and predicting.

        The dataframe passed to `fit` and `predict` will have a column with the
        specified name to be used as a regressor. When standardize='auto', the
        regressor will be standardized unless it is binary. The regression
        coefficient is given a prior with the specified scale parameter.
        Decreasing the prior scale will add additional regularization. If no
        prior scale is provided, self.holidays_prior_scale will be used.
        Mode can be specified as either 'additive' or 'multiplicative'. If not
        specified, self.seasonality_mode will be used. 'additive' means the
        effect of the regressor will be added to the trend, 'multiplicative'
        means it will multiply the trend.

        Parameters
        ----------
        name: string name of the regressor.
        prior_scale: optional float scale for the normal prior. If not
            provided, self.holidays_prior_scale will be used.
        standardize: optional, specify whether this regressor will be
            standardized prior to fitting. Can be 'auto' (standardize if not
            binary), True, or False.
        mode: optional, 'additive' or 'multiplicative'. Defaults to
            self.seasonality_mode.

        Returns
        -------
        The prophet object.
        """
        if self.history is not None:
            raise Exception(
                "Regressors must be added prior to model fitting.")
        self.validate_column_name(name, check_regressors=False)
        if prior_scale is None:
            prior_scale = float(self.holidays_prior_scale)
        if mode is None:
            mode = self.seasonality_mode
        if prior_scale <= 0:
            raise ValueError('Prior scale must be > 0')
        if mode not in ['additive', 'multiplicative']:
            raise ValueError("mode must be 'additive' or 'multiplicative'")
        self.extra_regressors[name] = {
            'prior_scale': prior_scale,
            'standardize': standardize,
            'mu': 0.,
            'std': 1.,
            'mode': mode,
        }
        return self

    def add_seasonality(
            self, name, period, fourier_order, prior_scale=None, mode=None
    ):
        """Add a seasonal component with specified period, number of Fourier
        components, and prior scale.

        Increasing the number of Fourier components allows the seasonality to
        change more quickly (at risk of overfitting). Default values for yearly
        and weekly seasonalities are 10 and 3 respectively.

        Increasing prior scale will allow this seasonality component more
        flexibility, decreasing will dampen it. If not provided, will use the
        seasonality_prior_scale provided on Prophet initialization (defaults
        to 10).

        Mode can be specified as either 'additive' or 'multiplicative'. If not
        specified, self.seasonality_mode will be used (defaults to additive).
        Additive means the seasonality will be added to the trend,
        multiplicative means it will multiply the trend.

        Parameters
        ----------
        name: string name of the seasonality component.
        period: float number of days in one period.
        fourier_order: int number of Fourier components to use.
        prior_scale: optional float prior scale for this component.
        mode: optional 'additive' or 'multiplicative'

        Returns
        -------
        The prophet object.
        """
        if self.history is not None:
            raise Exception(
                "Seasonality must be added prior to model fitting.")
        if name not in ['daily', 'weekly', 'yearly']:
            # Allow overwriting built-in seasonalities
            self.validate_column_name(name, check_seasonalities=False)
        if prior_scale is None:
            ps = self.seasonality_prior_scale
        else:
            ps = float(prior_scale)
        if ps <= 0:
            raise ValueError('Prior scale must be > 0')
        if mode is None:
            mode = self.seasonality_mode
        if mode not in ['additive', 'multiplicative']:
            raise ValueError("mode must be 'additive' or 'multiplicative'")
        self.seasonalities[name] = {
            'period': period,
            'fourier_order': fourier_order,
            'prior_scale': ps,
            'mode': mode,
        }
        return self

    def make_all_seasonality_features(self, df):
        """Dataframe with seasonality features.

        Includes seasonality features, holiday features, and added regressors.

        Parameters
        ----------
        df: pd.DataFrame with dates for computing seasonality features and any
            added regressors.

        Returns
        -------
        pd.DataFrame with regression features.
        list of prior scales for each column of the features dataframe.
        Dataframe with indicators for which regression components correspond to
            which columns.
        Dictionary with keys 'additive' and 'multiplicative' listing the
            component names for each mode of seasonality.
        """
        seasonal_features = []
        prior_scales = []
        modes = {'additive': [], 'multiplicative': []}

        # Seasonality features
        for name, props in self.seasonalities.items():
            features = self.make_seasonality_features(
                df['ds'],
                props['period'],
                props['fourier_order'],
                name,
            )
            seasonal_features.append(features)
            prior_scales.extend(
                [props['prior_scale']] * features.shape[1])
            modes[props['mode']].append(name)

        # Holiday features
        if self.holidays is not None or self.append_holidays is not None:
            features, holiday_priors, holiday_names = (
                self.make_holiday_features(df['ds'])
            )
            seasonal_features.append(features)
            prior_scales.extend(holiday_priors)
            modes[self.seasonality_mode].extend(holiday_names)

        # Additional regressors
        for name, props in self.extra_regressors.items():
            seasonal_features.append(pd.DataFrame(df[name]))
            prior_scales.append(props['prior_scale'])
            modes[props['mode']].append(name)

        # Dummy to prevent empty X
        if len(seasonal_features) == 0:
            seasonal_features.append(
                pd.DataFrame({'zeros': np.zeros(df.shape[0])}))
            prior_scales.append(1.)

        seasonal_features = pd.concat(seasonal_features, axis=1)
        component_cols, modes = self.regressor_column_matrix(
            seasonal_features, modes
        )
        return seasonal_features, prior_scales, component_cols, modes

    def regressor_column_matrix(self, seasonal_features, modes):
        """Dataframe indicating which columns of the feature matrix correspond
        to which seasonality/regressor components.

        Includes combination components, like 'additive_terms'. These
        combination components will be added to the 'modes' input.

        Parameters
        ----------
        seasonal_features: Constructed seasonal features dataframe
        modes: Dictionary with keys 'additive' and 'multiplicative' listing the
            component names for each mode of seasonality.

        Returns
        -------
        component_cols: A binary indicator dataframe with columns seasonal
            components and rows columns in seasonal_features. Entry is 1 if
            that columns is used in that component.
        modes: Updated input with combination components.
        """
        components = pd.DataFrame({
            'col': np.arange(seasonal_features.shape[1]),
            'component': [
                x.split('_delim_')[0] for x in seasonal_features.columns
            ],
        })
        # Add total for holidays
        if self.train_holiday_names is not None:
            components = self.add_group_component(
                components, 'holidays', self.train_holiday_names.unique())
        # Add totals additive and multiplicative components, and regressors
        for mode in ['additive', 'multiplicative']:
            components = self.add_group_component(
                components, mode + '_terms', modes[mode]
            )
            regressors_by_mode = [
                r for r, props in self.extra_regressors.items()
                if props['mode'] == mode
            ]
            components = self.add_group_component(
                components, 'extra_regressors_' + mode, regressors_by_mode)
            # Add combination components to modes
            modes[mode].append(mode + '_terms')
            modes[mode].append('extra_regressors_' + mode)
        # After all of the additive/multiplicative groups have been added,
        modes[self.seasonality_mode].append('holidays')
        # Convert to a binary matrix
        component_cols = pd.crosstab(
            components['col'], components['component'],
        ).sort_index(level='col')
        # Add columns for additive and multiplicative terms, if missing
        for name in ['additive_terms', 'multiplicative_terms']:
            if name not in component_cols:
                component_cols[name] = 0
        # Remove the placeholder
        component_cols.drop('zeros', axis=1, inplace=True, errors='ignore')
        # Validation
        if (
                max(component_cols['additive_terms']
                    + component_cols['multiplicative_terms']) > 1
        ):
            raise Exception('A bug occurred in seasonal components.')
        # Compare to the training, if set.
        if self.train_component_cols is not None:
            component_cols = component_cols[self.train_component_cols.columns]
            if not component_cols.equals(self.train_component_cols):
                raise Exception('A bug occurred in constructing regressors.')
        return component_cols, modes

    def add_group_component(self, components, name, group):
        """Adds a component with given name that contains all of the components
        in group.

        Parameters
        ----------
        components: Dataframe with components.
        name: Name of new group component.
        group: List of components that form the group.

        Returns
        -------
        Dataframe with components.
        """
        new_comp = components[components['component'].isin(set(group))].copy()
        group_cols = new_comp['col'].unique()
        if len(group_cols) > 0:
            new_comp = pd.DataFrame({'col': group_cols, 'component': name})
            components = components.append(new_comp)
        return components

    def parse_seasonality_args(self, name, arg, auto_disable, default_order):
        """Get number of fourier components for built-in seasonalities.

        Parameters
        ----------
        name: string name of the seasonality component.
        arg: 'auto', True, False, or number of fourier components as provided.
        auto_disable: bool if seasonality should be disabled when 'auto'.
        default_order: int default fourier order

        Returns
        -------
        Number of fourier components, or 0 for disabled.
        """
        if arg == 'auto':
            fourier_order = 0
            if name in self.seasonalities:
                logger.info(
                    'Found custom seasonality named "{name}", '
                    'disabling built-in {name} seasonality.'.format(name=name)
                )
            elif auto_disable:
                logger.info(
                    'Disabling {name} seasonality. Run prophet with '
                    '{name}_seasonality=True to override this.'.format(
                        name=name)
                )
            else:
                fourier_order = default_order
        elif arg is True:
            fourier_order = default_order
        elif arg is False:
            fourier_order = 0
        else:
            fourier_order = int(arg)
        return fourier_order

    def set_auto_seasonalities(self):
        """Set seasonalities that were left on auto.

        Turns on yearly seasonality if there is >=2 years of history.
        Turns on weekly seasonality if there is >=2 weeks of history, and the
        spacing between dates in the history is <7 days.
        Turns on daily seasonality if there is >=2 days of history, and the
        spacing between dates in the history is <1 day.
        """
        first = self.history['ds'].min()
        last = self.history['ds'].max()
        dt = self.history['ds'].diff()
        min_dt = dt.iloc[dt.nonzero()[0]].min()

        # Yearly seasonality
        yearly_disable = last - first < pd.Timedelta(days=730)
        fourier_order = self.parse_seasonality_args(
            'yearly', self.yearly_seasonality, yearly_disable, 10)
        if fourier_order > 0:
            self.seasonalities['yearly'] = {
                'period': 365.25,
                'fourier_order': fourier_order,
                'prior_scale': self.seasonality_prior_scale,
                'mode': self.seasonality_mode,
            }

        # Weekly seasonality
        weekly_disable = ((last - first < pd.Timedelta(weeks=2)) or
                          (min_dt >= pd.Timedelta(weeks=1)))
        fourier_order = self.parse_seasonality_args(
            'weekly', self.weekly_seasonality, weekly_disable, 3)
        if fourier_order > 0:
            self.seasonalities['weekly'] = {
                'period': 7,
                'fourier_order': fourier_order,
                'prior_scale': self.seasonality_prior_scale,
                'mode': self.seasonality_mode,
            }

        # Daily seasonality
        daily_disable = ((last - first < pd.Timedelta(days=2)) or
                         (min_dt >= pd.Timedelta(days=1)))
        fourier_order = self.parse_seasonality_args(
            'daily', self.daily_seasonality, daily_disable, 4)
        if fourier_order > 0:
            self.seasonalities['daily'] = {
                'period': 1,
                'fourier_order': fourier_order,
                'prior_scale': self.seasonality_prior_scale,
                'mode': self.seasonality_mode,
            }

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
        T = df['t'].iloc[i1] - df['t'].iloc[i0]
        k = (df['y_scaled'].iloc[i1] - df['y_scaled'].iloc[i0]) / T
        m = df['y_scaled'].iloc[i0] - k * df['t'].iloc[i0]
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
        T = df['t'].iloc[i1] - df['t'].iloc[i0]

        # Force valid values, in case y > cap or y < 0
        C0 = df['cap_scaled'].iloc[i0]
        C1 = df['cap_scaled'].iloc[i1]
        y0 = max(0.01 * C0, min(0.99 * C0, df['y_scaled'].iloc[i0]))
        y1 = max(0.01 * C1, min(0.99 * C1, df['y_scaled'].iloc[i1]))

        r0 = C0 / y0
        r1 = C1 / y1

        if abs(r0 - r1) <= 0.01:
            r0 = 1.05 * r0

        L0 = np.log(r0 - 1)
        L1 = np.log(r1 - 1)

        # Initialize the offset
        m = L0 * T / (L0 - L1)
        # And the rate
        k = (L0 - L1) / T
        return (k, m)

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
        if ('ds' not in df) or ('y' not in df):
            raise ValueError(
                "Dataframe must have columns 'ds' and 'y' with the dates and "
                "values respectively."
            )
        history = df[df['y'].notnull()].copy()
        if history.shape[0] < 2:
            raise ValueError('Dataframe has less than 2 non-NaN rows.')
        self.history_dates = pd.to_datetime(df['ds']).sort_values()

        history = self.setup_dataframe(history, initialize_scales=True)
        self.history = history
        self.set_auto_seasonalities()
        seasonal_features, prior_scales, component_cols, modes = (
            self.make_all_seasonality_features(history))
        self.train_component_cols = component_cols
        self.component_modes = modes

        self.set_changepoints()

        dat = {
            'T': history.shape[0],
            'K': seasonal_features.shape[1],
            'S': len(self.changepoints_t),
            'y': history['y_scaled'],
            't': history['t'],
            't_change': self.changepoints_t,
            'X': seasonal_features,
            'sigmas': prior_scales,
            'tau': self.changepoint_prior_scale,
            'trend_indicator': int(self.growth == 'logistic'),
            's_a': component_cols['additive_terms'],
            's_m': component_cols['multiplicative_terms'],
        }

        if self.growth == 'linear':
            dat['cap'] = np.zeros(self.history.shape[0])
            kinit = self.linear_growth_init(history)
        else:
            dat['cap'] = history['cap_scaled']
            kinit = self.logistic_growth_init(history)

        model = prophet_stan_model

        def stan_init():
            return {
                'k': kinit[0],
                'm': kinit[1],
                'delta': np.zeros(len(self.changepoints_t)),
                'beta': np.zeros(seasonal_features.shape[1]),
                'sigma_obs': 1,
            }

        if (
                (history['y'].min() == history['y'].max())
                and self.growth == 'linear'
        ):
            # Nothing to fit.
            self.params = stan_init()
            self.params['sigma_obs'] = 1e-9
            for par in self.params:
                self.params[par] = np.array([self.params[par]])
        elif self.mcmc_samples > 0:
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
            if df.shape[0] == 0:
                raise ValueError('Dataframe has no rows.')
            df = self.setup_dataframe(df.copy())

        df['trend'] = self.predict_trend(df)
        seasonal_components = self.predict_seasonal_components(df)
        intervals = self.predict_uncertainty(df)

        # Drop columns except ds, cap, floor, and trend
        cols = ['ds', 'trend']
        if 'cap' in df:
            cols.append('cap')
        if self.logistic_floor:
            cols.append('floor')
        # Add in forecast components
        df2 = pd.concat((df[cols], intervals, seasonal_components), axis=1)
        df2['yhat'] = (
                df2['trend'] * (1 + df2['multiplicative_terms'])
                + df2['additive_terms']
        )
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
                    * (1 - k_cum[i] / k_cum[i + 1])  # noqa W503
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

        return trend * self.y_scale + df['floor']

    def predict_seasonal_components(self, df):
        """Predict seasonality components, holidays, and added regressors.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Dataframe with seasonal components.
        """
        seasonal_features, _, component_cols, _ = (
            self.make_all_seasonality_features(df)
        )
        lower_p = 100 * (1.0 - self.interval_width) / 2
        upper_p = 100 * (1.0 + self.interval_width) / 2

        X = seasonal_features.values
        data = {}
        for component in component_cols.columns:
            beta_c = self.params['beta'] * component_cols[component].values

            comp = np.matmul(X, beta_c.transpose())
            if component in self.component_modes['additive']:
                comp *= self.y_scale
            data[component] = np.nanmean(comp, axis=1)
            data[component + '_lower'] = np.nanpercentile(
                comp, lower_p, axis=1,
            )
            data[component + '_upper'] = np.nanpercentile(
                comp, upper_p, axis=1,
            )
        return pd.DataFrame(data)

    def sample_posterior_predictive(self, df):
        """Prophet posterior predictive samples.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Dictionary with posterior predictive samples for the forecast yhat and
        for the trend component.
        """
        n_iterations = self.params['k'].shape[0]
        samp_per_iter = max(1, int(np.ceil(
            self.uncertainty_samples / float(n_iterations)
        )))

        # Generate seasonality features once so we can re-use them.
        seasonal_features, _, component_cols, _ = (
            self.make_all_seasonality_features(df)
        )

        sim_values = {'yhat': [], 'trend': []}
        for i in range(n_iterations):
            for _j in range(samp_per_iter):
                sim = self.sample_model(
                    df=df,
                    seasonal_features=seasonal_features,
                    iteration=i,
                    s_a=component_cols['additive_terms'],
                    s_m=component_cols['multiplicative_terms'],
                )
                for key in sim_values:
                    sim_values[key].append(sim[key])
        for k, v in sim_values.items():
            sim_values[k] = np.column_stack(v)
        return sim_values

    def predictive_samples(self, df):
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        df: Dataframe with dates for predictions (column ds), and capacity
            (column cap) if logistic growth.

        Returns
        -------
        Dictionary with keys "trend" and "yhat" containing
        posterior predictive samples for that component.
        """
        df = self.setup_dataframe(df.copy())
        sim_values = self.sample_posterior_predictive(df)
        return sim_values

    def predict_uncertainty(self, df):
        """Prediction intervals for yhat and trend.

        Parameters
        ----------
        df: Prediction dataframe.

        Returns
        -------
        Dataframe with uncertainty intervals.
        """
        sim_values = self.sample_posterior_predictive(df)

        lower_p = 100 * (1.0 - self.interval_width) / 2
        upper_p = 100 * (1.0 + self.interval_width) / 2

        series = {}
        for key in ['yhat', 'trend']:
            series['{}_lower'.format(key)] = np.nanpercentile(
                sim_values[key], lower_p, axis=1)
            series['{}_upper'.format(key)] = np.nanpercentile(
                sim_values[key], upper_p, axis=1)

        return pd.DataFrame(series)

    def sample_model(self, df, seasonal_features, iteration, s_a, s_m):
        """Simulate observations from the extrapolated generative model.

        Parameters
        ----------
        df: Prediction dataframe.
        seasonal_features: pd.DataFrame of seasonal features.
        iteration: Int sampling iteration to use parameters from.
        s_a: Indicator vector for additive components
        s_m: Indicator vector for multiplicative components

        Returns
        -------
        Dataframe with trend and yhat, each like df['t'].
        """
        trend = self.sample_predictive_trend(df, iteration)

        beta = self.params['beta'][iteration]
        Xb_a = np.matmul(seasonal_features.values, beta * s_a) * self.y_scale
        Xb_m = np.matmul(seasonal_features.values, beta * s_m)

        sigma = self.params['sigma_obs'][iteration]
        noise = np.random.normal(0, sigma, df.shape[0]) * self.y_scale

        return pd.DataFrame({
            'yhat': trend * (1 + Xb_m) + Xb_a + noise,
            'trend': trend
        })

    def sample_predictive_trend(self, df, iteration):
        """Simulate the trend using the extrapolated generative model.

        Parameters
        ----------
        df: Prediction dataframe.
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

        # New changepoints from a Poisson process with rate S on [1, T]
        if T > 1:
            S = len(self.changepoints_t)
            n_changes = np.random.poisson(S * (T - 1))
        else:
            n_changes = 0
        if n_changes > 0:
            changepoint_ts_new = 1 + np.random.rand(n_changes) * (T - 1)
            changepoint_ts_new.sort()
        else:
            changepoint_ts_new = []

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

        return trend * self.y_scale + df['floor']

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
        if self.history_dates is None:
            raise Exception('Model must be fit before this can be used.')
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
        return plot(
            m=self, fcst=fcst, ax=ax, uncertainty=uncertainty,
            plot_cap=plot_cap, xlabel=xlabel, ylabel=ylabel,
        )

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
        return plot_components(
            m=self, fcst=fcst, uncertainty=uncertainty, plot_cap=plot_cap,
            weekly_start=weekly_start, yearly_start=yearly_start,
        )

    def plot_forecast_component(
            self, fcst, name, ax=None, uncertainty=True, plot_cap=False):
        warnings.warn(
            'This method will be removed in the next version. '
            'Please use fbprophet.plot.plot_forecast_component. ',
            DeprecationWarning,
        )
        return plot_forecast_component(
            self, fcst=fcst, name=name, ax=ax, uncertainty=uncertainty,
            plot_cap=plot_cap,
        )

    def seasonality_plot_df(self, ds):
        warnings.warn(
            'This method will be removed in the next version. '
            'Please use fbprophet.plot.seasonality_plot_df. ',
            DeprecationWarning,
        )
        return seasonality_plot_df(self, ds=ds)

    def plot_weekly(self, ax=None, uncertainty=True, weekly_start=0):
        warnings.warn(
            'This method will be removed in the next version. '
            'Please use fbprophet.plot.plot_weekly. ',
            DeprecationWarning,
        )
        return plot_weekly(
            self, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start,
        )

    def plot_yearly(self, ax=None, uncertainty=True, yearly_start=0):
        warnings.warn(
            'This method will be removed in the next version. '
            'Please use fbprophet.plot.plot_yearly. ',
            DeprecationWarning,
        )
        return plot_yearly(
            self, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start,
        )

    def plot_seasonality(self, name, ax=None, uncertainty=True):
        warnings.warn(
            'This method will be removed in the next version. '
            'Please use fbprophet.plot.plot_seasonality. ',
            DeprecationWarning,
        )
        return plot_seasonality(
            self, name=name, ax=ax, uncertainty=uncertainty,
        )

    def copy(self, cutoff=None):
        warnings.warn(
            'This method will be removed in the next version. '
            'Please use fbprophet.diagnostics.prophet_copy. ',
            DeprecationWarning,
        )
        return prophet_copy(m=self, cutoff=cutoff)
