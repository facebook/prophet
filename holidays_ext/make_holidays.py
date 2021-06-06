# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from typing import List
import warnings

import pandas as pd

from holidays_ext import holidays_ext
import holidays


def get_holiday(
        country_list: List[str],
        years: List[int]) -> dict:
    """Looks up available holidays in a given country list and a given list of years.
    Primarily looks up holidays in ``holidays_ext``.
    If not in ``holidays_ext``, looks up in ``holidays``.
    If not in ``holidays``, raises ValueError.

    Parameters
    ----------
    country_list : `list` [`str`]
        A list of country names to look up holidays.
    years : `list` [`int`]
        A list of years to look up holidays.

    Returns
    -------
    holidays : `dict`
        A dictionary with keys equals the countries.
        The values are dictionaries with keys being datetimes and values being the name of holidays.
    """
    result = {}
    for country in country_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result[country] = getattr(holidays_ext, country)(years=years)
        except AttributeError:
            try:
                result[country] = getattr(holidays, country)(years=years)
            except AttributeError:
                raise AttributeError(f"Holidays in {country} are not currently supported!")
    return result


def make_holidays_df(
        country_list: List[str],
        years: List[int]) -> dict:
    """Generates a dataframe with holidays for a given country list and a list of years.

    Parameters
    ----------
    country_list : `list` [`str`]
        A list of country names to look up holidays.
    years : `list` [`int`]
        A list of years to look up holidays.

    Returns
    -------
    holidays_df : `pandas.DataFrame`
        A dataframe with the following columns:

            "ts" : `pandas.Timestamp`
                The timestamps.
            "holiday" : `str`
                The holiday name.
            "country" : `str`
                The country name.
            "country_holiday" : `str`
                The country and holiday, in case two countries have the same holiday.
    """
    holidays = get_holiday(
        country_list=country_list,
        years=years
    )
    dfs = []
    for country in holidays:
        temp_df = pd.DataFrame(holidays[country]).transpose().reset_index(drop=True)
        temp_df.columns = ["ts", "holiday"]
        temp_df["country"] = country
        temp_df["country_holiday"] = temp_df["country"] + "_" + temp_df["holiday"]
        temp_df["ts"] = pd.to_datetime(temp_df["ts"])
        dfs.append(temp_df)
    return pd.concat(dfs, axis=0)
