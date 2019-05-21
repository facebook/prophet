# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#' Return all possible holiday names of given country
#'
#' @param country.name Country name (character).
#'
#' @return A vector of all possible holiday names (unique) of given country.
#' @keywords internal
get_holiday_names <- function(country.name){
    holidays <- generated_holidays %>% 
      dplyr::filter(country == country.name) %>%
      dplyr::select(holiday) %>%
      unique()
  return(holidays$holiday)
}


#' Make dataframe of holidays for given years and countries
#'
#' @param years List of years for which to include holiday dates.
#' @param country.name Country name (character).
#'
#' @return Dataframe with 'ds' and 'holiday', which can directly feed
#'  to 'holidays' params in Prophet
#' @keywords internal
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
    dplyr::mutate(ds = as.Date(ds)) %>%
    data.frame
  return(holidays.df)
}
