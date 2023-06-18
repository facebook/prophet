# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import unicodedata

import pandas as pd
import numpy as np

from holidays import list_supported_countries
from prophet.make_holidays import make_holidays_df


def utf8_to_ascii(text: str) -> str:
    """Holidays often have utf-8 characters. These are not allowed in R package data (they generate a NOTE).
    TODO: revisit whether we want to do this lossy conversion.
    """
    ascii_text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")
    # Remove trailing empty brackets and spaces.
    ascii_text = re.sub(r"\(\)$", "", ascii_text).strip()

    # Check if anything converted
    if sum(1 for x in ascii_text if x not in [" ", "(", ")", ","]) == 0:
        return "FAILED_TO_PARSE"
    else:
        return ascii_text


def generate_holidays_df() -> pd.DataFrame:
    """Generate csv file of all possible holiday names, ds, and countries, year combination."""
    country_codes = set(list_supported_countries().keys())

    # For compatibility with Turkey as 'TU' cases.
    country_codes.add("TU")

    all_holidays = []
    for country_code in country_codes:
        df = make_holidays_df(
            year_list=np.arange(1995, 2045, 1).tolist(),
            country=country_code,
        )
        df["country"] = country_code
        all_holidays.append(df)

    generated_holidays = pd.concat(all_holidays, axis=0, ignore_index=True)
    generated_holidays["year"] = generated_holidays.ds.dt.year
    generated_holidays.sort_values(["country", "ds", "holiday"], inplace=True)

    # Convert to ASCII, and drop holidays that fail to convert
    generated_holidays["holiday"] = generated_holidays["holiday"].apply(utf8_to_ascii)
    failed_countries = generated_holidays.loc[
        generated_holidays["holiday"] == "FAILED_TO_PARSE", "country"
    ].unique()
    if len(failed_countries) > 0:
        print("Failed to convert UTF-8 holidays for:")
        print("\n".join(failed_countries))
    assert "FAILED_TO_PARSE" not in generated_holidays["holiday"].unique()
    return generated_holidays


if __name__ == "__main__":
    import argparse
    import pathlib

    if not pathlib.Path.cwd().stem == "python":
        raise RuntimeError("Run script from prophet/python directory")
    OUT_CSV_PATH = pathlib.Path(".") / ".." / "R/data-raw/generated_holidays.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", default=OUT_CSV_PATH)
    args = parser.parse_args()
    df = generate_holidays_df()
    df.to_csv(args.outfile, index=False)
