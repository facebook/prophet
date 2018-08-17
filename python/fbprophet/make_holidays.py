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
import fbprophet.hdays as hdays_part1
import holidays as hdays_part2


def make_holidays(year_list, country_list):
    """Make dataframe of holidays for given years and countries

    Parameters
    ----------
    year_list: a list of years
    country_list: a list of countries

    Returns
    -------
    Dataframe with 'ds' and 'holiday', which can directly feed
    to 'holidays' params in Prophet
    """

    if isinstance(country_list, str):
        country_list = [country_list]
    all_hdays = []
    for country in country_list:
        try:
            temp = getattr(hdays_part1, country)(years=year_list)
        except AttributeError:
            try:
                temp = getattr(hdays_part2, country)(years=year_list)
            except AttributeError:
                raise AttributeError(
                    "Holidays in {} are not currently supported!".format(country))
        temp_df = pd.DataFrame(list(temp.items()),
                               columns=['ds', 'holiday'])
        all_hdays.append(temp_df)
    res = pd.concat(all_hdays, axis=0, ignore_index=True)
    res.reset_index(inplace=True, drop=True)
    res['ds'] = pd.to_datetime(res['ds'])
    return (res)
