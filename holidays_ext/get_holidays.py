# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from datetime import datetime
import inspect
from typing import List, Optional
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


def get_holiday_df(
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
        temp_df = pd.DataFrame(holidays[country], index=[0]).transpose()
        temp_df.columns = ["holiday"]
        temp_df["ts"] = pd.to_datetime(temp_df.index)
        temp_df["country"] = country
        temp_df["country_holiday"] = temp_df["country"] + "_" + temp_df["holiday"]
        temp_df["ts"] = pd.to_datetime(temp_df["ts"])
        dfs.append(temp_df)
    return pd.concat(dfs, axis=0).reset_index(drop=True)[["ts", "country", "holiday", "country_holiday"]]


def get_available_holiday_lookup_countries(countries: Optional[List[str]] = None) -> List[str]:
    """Returns a list of available countries for holidays.

    Parameters
    ----------
    countries : `list` [`str`]
        A list of countries. If not provided, the function will return all available countries.
        If provided, the function only returns available countries in ``countries``.

    Returns
    -------
    available_countries : `list` [`str`]
        A list of available countries for holidays.
    """
    base_holidays = [name for name, obj in inspect.getmembers(holidays)
                     if inspect.isclass(obj) and obj.__module__ == holidays.__name__
                     and name != "HolidayBase"]
    ext_holidays = [name for name, obj in inspect.getmembers(holidays_ext)
                    if inspect.isclass(obj) and obj.__module__ == holidays_ext.__name__]
    all_countries = set(base_holidays + ext_holidays)

    if countries is None:
        return sorted(list(all_countries))

    subset_countries = [c for c in set(countries) if c in all_countries]
    return sorted(subset_countries)


def get_available_holidays_in_countries(
        countries : List[str],
        year_start : Optional[int] = None,
        year_end : Optional[int] = None) -> dict:
    """For a list of countries and a range of years, returns all available holidays in these countries.

    Parameters
    ----------
    countries : `list` [`str`]
        A list of countries.
    year_start : `int` or None, default None
        When to start looking for holidays.
        If None, will start from 1985.
    year_end : `int` or None, default None
        When to end looking for holidays.
        If None, will end at the current year.

    Returns
    -------
    holidays : `dict`
        A dictionary with keys being the countries and values being a list of holidays.
    """
    if year_start is None:
        year_start = 1985
    if year_end is None:
        current_year = datetime.now().year
        if current_year < year_start:
            year_end = year_start
        else:
            year_end = current_year

    country_holidays = get_holiday(
        country_list=countries,
        years=[x for x in range(year_start, year_end + 1)]
    )
    return {country: sorted(list(set(holiday.values()))) for country, holiday in country_holidays.items()}


def get_available_holidays_across_countries(
        countries : List[str],
        year_start : Optional[int] = None,
        year_end : Optional[int] = None) -> List[str]:
    """For a list of countries and a range of years, returns a list of holidays that occurs in
     any of these countries.

    Parameters
    ----------
    countries : `list` [`str`]
        A list of countries.
    year_start : `int` or None, default None
        When to start looking for holidays.
        If None, will start from 1985.
    year_end : `int` or None, default None
        When to end looking for holidays.
        If None, will end at the current year.

    Returns
    -------
    holidays : `list`
        A list of holidays that occur in any of the countries.
    """
    country_holidays = get_available_holidays_in_countries(
        countries=countries,
        year_start=year_start,
        year_end=year_end
    )
    return sorted(list({h for h_list in country_holidays.values() for h in h_list}))
