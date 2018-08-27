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

import pandas as pd
import numpy as np
import warnings
import holidays as hdays_part1
import fbprophet.hdays as hdays_part2


def get_holiday_names(country):
    """Return all possible holiday names of given country

    Parameters
    ----------
    country: country name

    Returns
    ------- a
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """
    years = np.arange(1995, 2045)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            holiday_names = getattr(hdays_part2, country)(years=years).values()
    except AttributeError:
        try:
            holiday_names = getattr(hdays_part1, country)(years=years).values()
        except AttributeError:
            raise AttributeError(
                "Holidays in {} are not currently supported!".format(country))
    return set(holiday_names)


def make_holidays_df(year_list, country):
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
        holidays = getattr(hdays_part2, country)(years=year_list)
    except AttributeError:
        try:
            holidays = getattr(hdays_part1, country)(years=year_list)
        except AttributeError:
            raise AttributeError(
                "Holidays in {} are not currently supported!".format(country))
    holidays_df = pd.DataFrame(list(holidays.items()), columns=['ds', 'holiday'])
    holidays_df.reset_index(inplace=True, drop=True)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    return (holidays_df)
