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
#' @param df Dataframe with historical data.
#' @param horizon timediff forecast horizon.
#' @param initial timediff initial window.
#' @param period timediff Simulated forecasts are done with this period.
#'
#' @return Array of datetimes.
#'
#' @keywords internal
generate_cutoffs <- function(df, horizon, initial, period) {
  # Last cutoff is (latest date in data) - (horizon).
  cutoff <- max(df$ds) - horizon
  tzone <- attr(cutoff, "tzone")  # Timezone is wiped by putting in array
  result <- c(cutoff)
  while (result[length(result)] >= min(df$ds) + initial) {
    cutoff <- cutoff - period
    # If data does not exist in data range (cutoff, cutoff + horizon]
    if (!any((df$ds > cutoff) & (df$ds <= cutoff + horizon))) {
        # Next cutoff point is 'closest date before cutoff in data - horizon'
        closest.date <- max(df$ds[df$ds <= cutoff])
        cutoff <- closest.date - horizon
    }
    result <- c(result, cutoff)
  }
  result <- utils::head(result, -1)
  if (length(result) == 0) {
    stop(paste(
      'Less data than horizon after initial window.',
      'Make horizon or initial shorter.'
    ))
  }
  # Reset timezones
  attr(result, "tzone") <- tzone
  message(paste(
    'Making', length(result), 'forecasts with cutoffs between',
    result[length(result)], 'and', result[1]
  ))
  return(rev(result))
}

#' Cross-validation for time series.
#'
#' Computes forecasts from historical cutoff points. Beginning from
#' (end - horizon), works backwards making cutoffs with a spacing of period
#' until initial is reached.
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
  df <- model$history
  te <- max(df$ds)
  ts <- min(df$ds)
  if (is.null(period)) {
    period <- 0.5 * horizon
  }
  if (is.null(initial)) {
    initial <- 3 * horizon
  }
  horizon.dt <- as.difftime(horizon, units = units)
  initial.dt <- as.difftime(initial, units = units)
  period.dt <- as.difftime(period, units = units)

  cutoffs <- generate_cutoffs(df, horizon.dt, initial.dt, period.dt)
  predicts <- data.frame()
  for (i in 1:length(cutoffs)) {
    cutoff <- cutoffs[i]
    # Copy the model
    m <- prophet_copy(model, cutoff)
    # Train model
    history.c <- dplyr::filter(df, ds <= cutoff)
    if (nrow(history.c) < 2) {
      stop('Less than two datapoints before cutoff. Increase initial window.')
    }
    m <- fit.prophet(m, history.c)
    # Calculate yhat
    df.predict <- dplyr::filter(df, ds > cutoff, ds <= cutoff + horizon.dt)
    # Get the columns for the future dataframe
    columns <- 'ds'
    if (m$growth == 'logistic') {
      columns <- c(columns, 'cap')
      if (m$logistic.floor) {
        columns <- c(columns, 'floor')
      }
    }
    columns <- c(columns, names(m$extra_regressors))
    future <- df.predict[columns]
    yhat <- stats::predict(m, future)
    # Merge yhat, y, and cutoff.
    df.c <- dplyr::inner_join(df.predict, yhat, by = "ds")
    df.c <- dplyr::select(df.c, ds, y, yhat, yhat_lower, yhat_upper)
    df.c$cutoff <- cutoff
    predicts <- rbind(predicts, df.c)
  }
  return(predicts)
}

#' Copy Prophet object.
#'
#' @param m Prophet model object.
#' @param cutoff Date, possibly as string. Changepoints are only retained if
#'  changepoints <= cutoff.
#'
#' @return An unfitted Prophet model object with the same parameters as the
#'  input model.
#'
#' @keywords internal
prophet_copy <- function(m, cutoff = NULL) {
  if (is.null(m$history)) {
    stop("This is for copying a fitted Prophet object.")
  }

  if (m$specified.changepoints) {
    changepoints <- m$changepoints
    if (!is.null(cutoff)) {
      cutoff <- set_date(cutoff)
      changepoints <- changepoints[changepoints <= cutoff]
    }
  } else {
    changepoints <- NULL
  }
  # Auto seasonalities are set to FALSE because they are already set in
  # m$seasonalities.
  m2 <- prophet(
    growth = m$growth,
    changepoints = changepoints,
    n.changepoints = m$n.changepoints,
    changepoint.range = m$changepoint.range,
    yearly.seasonality = FALSE,
    weekly.seasonality = FALSE,
    daily.seasonality = FALSE,
    holidays = m$holidays,
    seasonality.mode = m$seasonality.mode,
    seasonality.prior.scale = m$seasonality.prior.scale,
    changepoint.prior.scale = m$changepoint.prior.scale,
    holidays.prior.scale = m$holidays.prior.scale,
    mcmc.samples = m$mcmc.samples,
    interval.width = m$interval.width,
    uncertainty.samples = m$uncertainty.samples,
    fit = FALSE
  )
  m2$extra_regressors <- m$extra_regressors
  m2$seasonalities <- m$seasonalities
  return(m2)
}

#' Compute performance metrics from cross-validation results.
#'
#' Computes a suite of performance metrics on the output of cross-validation.
#' By default the following metrics are included:
#' 'mse': mean squared error
#' 'rmse': root mean squared error
#' 'mae': mean absolute error
#' 'mape': mean percent error
#' 'coverage': coverage of the upper and lower intervals
#'
#' A subset of these can be specified by passing a list of names as the
#' `metrics` argument.
#'
#' Metrics are calculated over a rolling window of cross validation
#' predictions, after sorting by horizon. The size of that window (number of
#' simulated forecast points) is determined by the rolling_window argument,
#' which specifies a proportion of simulated forecast points to include in
#' each window. rolling_window=0 will compute it separately for each simulated
#' forecast point (i.e., 'mse' will actually be squared error with no mean).
#' The default of rolling_window=0.1 will use 10% of the rows in df in each
#' window. rolling_window=1 will compute the metric across all simulated
#' forecast points. The results are set to the right edge of the window.
#'
#' The output is a dataframe containing column 'horizon' along with columns
#' for each of the metrics computed.
#'
#' @param df The dataframe returned by cross_validation.
#' @param metrics An array of performance metrics to compute. If not provided,
#'  will use c('mse', 'rmse', 'mae', 'mape', 'coverage').
#' @param rolling_window Proportion of data to use in each rolling window for
#'  computing the metrics. Should be in [0, 1].
#'
#' @return A dataframe with a column for each metric, and column 'horizon'.
#'
#' @export
performance_metrics <- function(df, metrics = NULL, rolling_window = 0.1) {
  valid_metrics <- c('mse', 'rmse', 'mae', 'mape', 'coverage')
  if (is.null(metrics)) {
    metrics <- valid_metrics
  }
  if (length(metrics) != length(unique(metrics))) {
    stop('Input metrics must be an array of unique values.')
  }
  if (!all(metrics %in% valid_metrics)) {
    stop(
      paste('Valid values for metrics are:', paste(metrics, collapse = ", "))
    )
  }
  df_m <- df
  df_m$horizon <- df_m$ds - df_m$cutoff
  df_m <- df_m[order(df_m$horizon),]
  # Window size
  w <- as.integer(rolling_window * nrow(df_m))
  w <- max(w, 1)
  w <- min(w, nrow(df_m))
  cols <- c('horizon')
  for (metric in metrics) {
    df_m[[metric]] <- get(metric)(df_m, w)
    cols <- c(cols, metric)
  }
  df_m <- df_m[cols]
  return(stats::na.omit(df_m))
}

#' Compute a rolling mean of x
#'
#' Right-aligned. Padded with NAs on the front so the output is the same
#' size as x.
#'
#' @param x Array.
#' @param w Integer window size (number of elements).
#'
#' @return Rolling mean of x with window size w.
#'
#' @keywords internal
rolling_mean <- function(x, w) {
  s <- cumsum(c(0, x))
  prefix <- rep(NA, w - 1)
  return(c(prefix, (s[(w + 1):length(s)] - s[1:(length(s) - w)]) / w))
}

# The functions below specify performance metrics for cross-validation results.
# Each takes as input the output of cross_validation, and returns the statistic
# as an array, given a window size for rolling aggregation.

#' Mean squared error
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of mean squared errors.
#'
#' @keywords internal
mse <- function(df, w) {
  se <- (df$y - df$yhat) ** 2
  return(rolling_mean(se, w))
}

#' Root mean squared error
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of root mean squared errors.
#'
#' @keywords internal
rmse <- function(df, w) {
  return(sqrt(mse(df, w)))
}

#' Mean absolute error
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of mean absolute errors.
#'
#' @keywords internal
mae <- function(df, w) {
  ae <- abs(df$y - df$yhat)
  return(rolling_mean(ae, w))
}

#' Mean absolute percent error
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of mean absolute percent errors.
#'
#' @keywords internal
mape <- function(df, w) {
  ape <- abs((df$y - df$yhat) / df$y)
  return(rolling_mean(ape, w))
}

#' Coverage
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of coverages
#'
#' @keywords internal
coverage <- function(df, w) {
  is_covered <- (df$y >= df$yhat_lower) & (df$y <= df$yhat_upper)
  return(rolling_mean(is_covered, w))
}
