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
  country.holidays = generated_holidays %>%
    dplyr::filter(country == country.name)
  max.year <- max(country.holidays$year)
  min.year <- min(country.holidays$year)
  if (max(years) > max.year ||  min(years) < min.year){
    warning.msg = paste("Holidays for", country.name, "are only supported from", min.year, 
                        "to", max.year)
    warning(warning.msg)
  }
  holidays.df <- country.holidays %>%
    dplyr::filter(year %in% years) %>%
    dplyr::select(ds, holiday) %>%
    data.frame
  return(holidays.df)
}