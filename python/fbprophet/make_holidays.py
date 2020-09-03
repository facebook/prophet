# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import pandas as pd

import fbprophet.hdays as hdays_part2
import holidays as hdays_part1


def get_holiday_names(country):
    """Return all possible holiday names of given country

    Parameters
    ----------
    country: country name

    Returns
    -------
    A set of all possible holiday names of given country
    """
    years = np.arange(1995, 2045)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holiday_names = getattr(hdays_part2, country)(years=years).values()
    except AttributeError:
        try:
            holiday_names = getattr(hdays_part1, country)(years=years).values()
        except AttributeError as e:
            raise AttributeError(
                "Holidays in {} are not currently supported!".format(country)) from e
    return set(holiday_names)


def make_holidays_df(year_list, country, province=None, state=None):
    """Make dataframe of holidays for given years and countries

    Parameters
    ----------
    year_list: a list of years
    country: country name

    Returns
    -------
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """
    try:
        holidays = getattr(hdays_part2, country)(years=year_list, expand=False)
    except AttributeError:
        try:
            holidays = getattr(hdays_part1, country)(prov=province, state=state, years=year_list, expand=False)
        except AttributeError as e:
            raise AttributeError(
                "Holidays in {} are not currently supported!".format(country)) from e
    holidays_df = pd.DataFrame([(date, holidays.get_list(date)) for date in holidays], columns=['ds', 'holiday'])
    holidays_df = holidays_df.explode('holiday')
    holidays_df.reset_index(inplace=True, drop=True)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return (holidays_df)
