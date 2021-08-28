# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import unicodedata
import warnings

import pandas as pd
import numpy as np

import holidays as hdays_part1
import prophet.hdays as hdays_part2
from prophet.make_holidays import make_holidays_df


def utf8_to_ascii(text):
    """Holidays often have utf-8 characters. These are not allowed in R
    package data (they generate a NOTE).
    TODO: revisit whether we want to do this lossy conversion.
    """
    ascii_text = (
        unicodedata.normalize('NFD', text)
        .encode('ascii', 'ignore')
        .decode('ascii')
        .strip()
    )
    # Check if anything converted
    if sum(1 for x in ascii_text if x not in [' ', '(', ')', ',']) == 0:
        return 'FAILED_TO_PARSE'
    else:
        return ascii_text


def generate_holidays_file():
    """Generate csv file of all possible holiday names, ds,
     and countries, year combination
    """
    year_list = np.arange(1995, 2045, 1).tolist()
    all_holidays = []
    # class names in holiday packages which are not countries
    # Also cut out countries with utf-8 holidays that don't parse to ascii
    class_to_exclude = {'rd', 'BY', 'BG', 'JP', 'RS', 'UA', 'KR'}

    class_list2 = inspect.getmembers(hdays_part2, inspect.isclass)
    country_set = {name for name in list(zip(*class_list2))[0] if len(name) == 2}
    class_list1 = inspect.getmembers(hdays_part1, inspect.isclass)
    country_set1 = {name for name in list(zip(*class_list1))[0] if len(name) == 2}
    country_set.update(country_set1)
    country_set -= class_to_exclude

    for country in country_set:
        df = make_holidays_df(year_list=year_list, country=country)
        df['country'] = country
        all_holidays.append(df)

    generated_holidays = pd.concat(all_holidays, axis=0, ignore_index=True)
    generated_holidays['year'] = generated_holidays.ds.dt.year
    generated_holidays.sort_values(['country', 'ds', 'holiday'], inplace=True)

    # Convert to ASCII, and drop holidays that fail to convert
    generated_holidays['holiday'] = generated_holidays['holiday'].apply(utf8_to_ascii)
    assert 'FAILED_TO_PARSE' not in generated_holidays['holiday'].unique()
    generated_holidays.to_csv("../R/data-raw/generated_holidays.csv", index=False)


if __name__ == "__main__":
    # execute only if run as a script
    generate_holidays_file()
