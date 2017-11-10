## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.

## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant
## of patent rights can be found in the PATENTS file in the same directory.

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c(
  "ds", "y", "cap", "yhat", "yhat_lower", "yhat_upper"))

#' Generate cutoff dates
#'
#' @param df Dataframe with historical data
#' @param horizon timediff forecast horizon
#' @param k integer number of forecast points
#' @param period timediff Simulated forecasts are done with this period.
#'
#' @return Array of datetimes
#'
#' @keywords internal
generate_cutoffs <- function(df, horizon, k, period) {
  # Last cutoff is (latest date in data) - (horizon).
  cutoff <- max(df$ds) - horizon
  if (cutoff < min(df$ds)) {
    stop('Less data than horizon.')
  }
  tzone <- attr(cutoff, "tzone")  # Timezone is wiped by putting in array
  result <- c(cutoff)
  if (k > 1) {
    for (i in 2:k) {
      cutoff <- cutoff - period
      # If data does not exist in data range (cutoff, cutoff + horizon]
      if (!any((df$ds > cutoff) & (df$ds <= cutoff + horizon))) {
        # Next cutoff point is 'closest date before cutoff in data - horizon'
        closest.date <- max(df$ds[df$ds <= cutoff])
        cutoff <- closest.date - horizon
      }
      if (cutoff < min(df$ds)) {
        warning('Not enough data for requested number of cutoffs! Using ', i)
        break
      }
      result <- c(result, cutoff)
    }
  }
  # Reset timezones
  attr(result, "tzone") <- tzone
  return(rev(result))
}

#' Simulated historical forecasts.
#'
#' Make forecasts from k historical cutoff points, working backwards from
#' (end - horizon) with a spacing of period between each cutoff.
#'
#' @param model Fitted Prophet model.
#' @param horizon Integer size of the horizon
#' @param units String unit of the horizon, e.g., "days", "secs".
#' @param k integer number of forecast points
#' @param period Integer amount of time between cutoff dates. Same units as
#'  horizon. If not provided, will use 0.5 * horizon.
#'
#' @return A dataframe with the forecast, actual value, and cutoff date.
#'
#' @export
simulated_historical_forecasts <- function(model, horizon, units, k,
                                           period = NULL) {
  df <- model$history
  horizon <- as.difftime(horizon, units = units)
  if (is.null(period)) {
    period <- horizon / 2
  } else {
    period <- as.difftime(period, units = units)
  }
  cutoffs <- generate_cutoffs(df, horizon, k, period)
  predicts <- data.frame()
  for (i in 1:length(cutoffs)) {
    cutoff <- cutoffs[i]
    # Copy the model
    m <- prophet_copy(model, cutoff)
    # Train model
    history.c <- dplyr::filter(df, ds <= cutoff)
    m <- fit.prophet(m, history.c)
    # Calculate yhat
    df.predict <- dplyr::filter(df, ds > cutoff, ds <= cutoff + horizon)
    # Get the columns for the future dataframe
    columns <- c('ds')
    if (m$growth == 'logistic') {
      columns <- c(columns, 'cap')
      if (m$logistic.floor) {
        columns <- c(columns, 'floor')
      }
    }
    columns <- c(columns, names(m$extra_regressors))
    future <- df[columns]
    yhat <- stats::predict(m, future)
    # Merge yhat, y, and cutoff.
    df.c <- dplyr::inner_join(df.predict, yhat, by = "ds")
    df.c <- dplyr::select(df.c, ds, y, yhat, yhat_lower, yhat_upper)
    df.c$cutoff <- cutoff
    predicts <- rbind(predicts, df.c)
  }
  return(predicts)
}

#' Cross-validation for time series.
#'
#' Computes forecasts from historical cutoff points. Beginning from initial,
#' makes cutoffs with a spacing of period up to (end - horizon).
#'
#' When period is equal to the time interval of the data, this is the
#' technique described in https://robjhyndman.com/hyndsight/tscv/ .
#'
#' @param model Fitted Prophet model.
#' @param horizon Integer size of the horizon
#' @param units String unit of the horizon, e.g., "days", "secs".
#' @param period Integer amount of time between cutoff dates. Same units as
#'  horizon. If not provided, 0.5 * horizon is used.
#' @param initial Integer size of the first training period. If not provided,
#'  3 * horizon is used. Same units as horizon.
#'
#' @return A dataframe with the forecast, actual value, and cutoff date.
#'
#' @export
cross_validation <- function(
    model, horizon, units, period = NULL, initial = NULL) {
  te <- max(model$history$ds)
  ts <- min(model$history$ds)
  if (is.null(period)) {
    period <- 0.5 * horizon
  }
  if (is.null(initial)) {
    initial <- 3 * horizon
  }
  horizon.dt <- as.difftime(horizon, units = units)
  initial.dt <- as.difftime(initial, units = units)
  period.dt <- as.difftime(period, units = units)
  k <- ceiling(
    as.double((te - horizon.dt) - (ts + initial.dt), units='secs') /
    as.double(period.dt, units = 'secs')
  )
  if (k < 1) {
    stop('Not enough data for specified horizon, period, and initial.')
  }
  return(simulated_historical_forecasts(model, horizon, units, k, period))
}
