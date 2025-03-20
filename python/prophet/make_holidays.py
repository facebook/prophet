# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

import holidays


def get_country_holidays_class(country):
    """Get class for a supported country.

    Parameters
    ----------
    country: country code

    Returns
    -------
    A valid country holidays class
    """
    substitutions = {
        "TU": "TR",  # For compatibility with Turkey as 'TU' cases.
    }

    country = substitutions.get(country, country)
    if not hasattr(holidays, country):
        raise AttributeError(f"Holidays in {country} are not currently supported!")

    return getattr(holidays, country)


def get_holiday_names(country):
    """Return all possible holiday names of given country

    Parameters
    ----------
    country: country name

    Returns
    -------
    A set of all possible holiday names of given country
    """
    country_holidays = get_country_holidays_class(country)
    return set(country_holidays(language="en_US", years=np.arange(1995, 2045)).values())


def make_holidays_df(year_list, country, province=None, state=None):
    """Make dataframe of holidays for given years and countries

    Parameters
    ----------
    year_list: a list of years
    country: country name
    province: province name

    Returns
    -------
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """
    country_holidays = get_country_holidays_class(country)
    holidays = country_holidays(expand=False, language="en_US", subdiv=province, years=year_list)

    holidays_df = pd.DataFrame(
        [(date, holidays.get_list(date)) for date in holidays],
        columns=["ds", "holiday"],
    )
    holidays_df = holidays_df.explode("holiday")
    holidays_df.reset_index(inplace=True, drop=True)
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

    return holidays_df
