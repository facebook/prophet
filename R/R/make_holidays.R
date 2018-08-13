## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.

## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant
## of patent rights can be found in the PATENTS file in the same directory.


#' Return all possible holiday names of given country
#'
#' @param country.name Country name (character).
#'
#' @return A vector of all possible holiday names (unique) of given country.
#' @export
get_holiday_names <- function(country.name){
    holidays <- generated_holidays %>% 
      dplyr::filter(country == country.name) %>%
      dplyr::select(holiday) %>%
      unique()
  return(holidays$holiday)
}


#' Make dataframe of holidays for given years and countries
#'
#' @param country.name Country name (character).
#'
#' @return Dataframe with 'ds' and 'holiday', which can directly feed
#'  to 'holidays' params in Prophet
#' @export
make_holidays_df <- function(years, country.name){
  max.year <- max(generated_holidays$year)
  min.year <- min(generated_holidays$year)
  if (country.name == 'Indonesia' || country.name == 'ID'){
    warning("We only support Nyepi holiday from 2009 to 2019")
  }
  if (country.name == 'Thailand' || country.name == 'TH'){
    warning("We only support Diwali and Holi holidays from 2010 to 2025")
  }
  if (country.name == 'India' || country.name == 'IN'){
    warning.msg = "We only support Asalha Puja holiday from 2006 to 2025 and \
                    Vassa holiday from 2006 to 2020"
    warning(warning.msg)
  }
  if (max(years) > max.year ||  min(years) < min.year){
    warning.msg = paste("We only support holidays from year", min.year, 
                        "to year", max.year)
    warning(warning.msg)
  }
  holidays.df <- generated_holidays %>%
    dplyr::filter(country == country.name, year %in% years) %>%
    dplyr::select(ds, holiday) %>%
    data.frame
  return(holidays.df)
}