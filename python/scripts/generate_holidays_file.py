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
import inspect


def generate_holidays_file():
    """Generate csv file of all possible holiday names, ds,
     and countries, year combination
    """
    years = np.arange(1995, 2045, 1)
    all_holidays = []
    # class names in holiday packages which are not countries
    class_to_exclude = set(['rd', 'datetime', 'date', 'HolidayBase', 'Calendar',
                            'LunarDate', 'timedelta', 'date'])

    class_list2 = inspect.getmembers(hdays_part2, inspect.isclass)
    country_set2 = set(list(zip(*class_list2))[0])
    country_set2 -= class_to_exclude
    for country in country_set2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp = getattr(hdays_part2, country)(years=years)
        temp_df = pd.DataFrame(list(temp.items()),
                                columns=['ds', 'holiday'])
        temp_df['country'] = country
        all_holidays.append(temp_df)

    class_list1 = inspect.getmembers(hdays_part1, inspect.isclass)
    country_set1 = set(list(zip(*class_list1))[0])
    country_set1 -= class_to_exclude
    # Avoid overwrting holidays get from hdays_part2
    country_set1 -= country_set2
    for country in country_set1:
        temp = getattr(hdays_part1, country)(years=years)
        temp_df = pd.DataFrame(list(temp.items()),
                                columns=['ds', 'holiday'])
        temp_df['country'] = country
        all_holidays.append(temp_df)

    generated_holidays = pd.concat(all_holidays, axis=0, ignore_index=True)
    generated_holidays['year'] = generated_holidays.ds.apply(lambda x: x.year)
    generated_holidays.to_csv("../R/data-raw/generated_holidays.csv")


if __name__ == "__main__":
    # execute only if run as a script
    generate_holidays_file()
