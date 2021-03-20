# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c(
  "ds", "y", "cap", "yhat", "yhat_lower", "yhat_upper", "size"))

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
        if (cutoff > min(df$ds)) {
          closest.date <- max(df$ds[df$ds <= cutoff])
          cutoff <- closest.date - horizon
        }
        # else no data left, leave cutoff as is, it will be dropped.
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
#' Computes forecasts from historical cutoff points which user can input.If
#' not provided, these are computed beginning from (end - horizon), and working
#' backwards making cutoffs with a spacing of period until initial is reached.
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
#' @param cutoffs Vector of cutoff dates to be used during
#'  cross-validtation. If not provided works beginning from (end - horizon),
#'  works backwards making cutoffs with a spacing of period until initial is
#'  reached.
#'
#' @return A dataframe with the forecast, actual value, and cutoff date.
#'
#' @export
cross_validation <- function(
    model, horizon, units, period = NULL, initial = NULL, cutoffs=NULL) {
  df <- model$history
  horizon.dt <- as.difftime(horizon, units = units)

  predict_columns <- c('ds', 'yhat')
  if (model$uncertainty.samples){
    predict_columns <- append(predict_columns, c('yhat_lower', 'yhat_upper'))
  }
  # Identify largest seasonality period
  period.max <- 0
  for (s in model$seasonalities) {
    period.max <- max(period.max, s$period)
  }
  seasonality.dt <- as.difftime(period.max, units = 'days')

  if (is.null(cutoffs)){

    # Set period
    if (is.null(period)) {
      period <- 0.5 * horizon
    }
    period.dt <- as.difftime(period, units = units)
    # Set initial
    if (is.null(initial)) {
      initial.dt <- max(
        as.difftime(3 * horizon, units = units),
        seasonality.dt
      )
    }else {
      initial.dt <- as.difftime(initial, units = units)
    }
    cutoffs <- generate_cutoffs(df, horizon.dt, initial.dt, period.dt)
  }else{
    cutoffs <- set_date(ds=cutoffs)
    # Validation
    if (min(cutoffs) <= min(df$ds)) {
      stop('Minimum cutoff value is not strictly greater than min date in history')
    }
    end_date_minus_horizon <- max(df$ds) - horizon.dt
    if (max(cutoffs) > end_date_minus_horizon) {
      stop('Maximum cutoff value is greater than end date minus horizon')
    }
    initial.dt <- cutoffs[1] - min(df$ds)
  }

  # Check if the initial window  (that is, the amount of time between the
  # start of the history and the first cutoff) is less than the
  # maximum seasonality period
  if (initial.dt < seasonality.dt) {
    warning(paste0('Seasonality has period of ', period.max, ' days which ',
      'is larger than initial window. Consider increasing initial.'))
  }

  predicts <- data.frame()
  for (i in 1:length(cutoffs)) {
    df.c <- single_cutoff_forecast(df, model, cutoffs[i], horizon.dt, predict_columns)
    predicts <- rbind(predicts, df.c)
  }
  return(predicts)
}

#' Forecast for a single cutoff.

#' Used in cross_validation function when evaluating for multiple cutoffs.
#'
#' @param df Dataframe with history for cutoff.
#' @param model Prophet model object.
#' @param cutoff Datetime of cutoff.
#' @param horizon.dt timediff forecast horizon.
#' @param predict_columns Array of names of columns to be returned in output.
#'
#' @return Dataframe with forecast, actual value, and cutoff.
#'
#' @keywords internal
single_cutoff_forecast <- function(df, model, cutoff, horizon.dt, predict_columns){
  m <- prophet_copy(model, cutoff)
  # Train model
  history.c <- dplyr::filter(df, ds <= cutoff)
  if (nrow(history.c) < 2) {
    stop('Less than two datapoints before cutoff. Increase initial window.')
  }
  fit.args <- c(list(m=m, df=history.c), model$fit.kwargs)
  m <- do.call(fit.prophet, fit.args)
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
  for (name in names(m$seasonalities)) {
    condition.name = m$seasonalities[[name]]$condition.name
    if (!is.null(condition.name)) {
      columns <- c(columns, condition.name)
    }
  }
  future <- df.predict[columns]
  yhat <- stats::predict(m, future)
  # Merge yhat, y, and cutoff.
  df.c <- dplyr::inner_join(df.predict, yhat[predict_columns], by = "ds")
  df.c <- df.c[c(predict_columns, "y")]
  df.c <- dplyr::select(df.c, y, predict_columns)
  df.c$cutoff <- cutoff
  return(df.c)
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
      last_history_date <- max(m$history$ds[m$history$ds <= cutoff])
      changepoints <- changepoints[changepoints < last_history_date]
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
  m2$country_holidays <- m$country_holidays
  return(m2)
}

#' Compute performance metrics from cross-validation results.
#'
#' Computes a suite of performance metrics on the output of cross-validation.
#' By default the following metrics are included:
#' 'mse': mean squared error,
#' 'rmse': root mean squared error,
#' 'mae': mean absolute error,
#' 'mape': mean percent error,
#' 'mdape': median percent error,
#' 'smape': symmetric mean absolute percentage error,
#' 'coverage': coverage of the upper and lower intervals
#'
#' A subset of these can be specified by passing a list of names as the
#' `metrics` argument.
#'
#' Metrics are calculated over a rolling window of cross validation
#' predictions, after sorting by horizon. Averaging is first done within each
#' value of the horizon, and then across horizons as needed to reach the
#' window size. The size of that window (number of simulated forecast points)
#' is determined by the rolling_window argument, which specifies a proportion
#' of simulated forecast points to include in each window. rolling_window=0
#' will compute it separately for each horizon. The default of
#' rolling_window=0.1 will use 10% of the rows in df in each window.
#' rolling_window=1 will compute the metric across all simulated forecast
#' points. The results are set to the right edge of the window.
#'
#' If rolling_window < 0, then metrics are computed at each datapoint with no
#' averaging (i.e., 'mse' will actually be squared error with no mean).
#'
#' The output is a dataframe containing column 'horizon' along with columns
#' for each of the metrics computed.
#'
#' @param df The dataframe returned by cross_validation.
#' @param metrics An array of performance metrics to compute. If not provided,
#'  will use c('mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage').
#' @param rolling_window Proportion of data to use in each rolling window for
#'  computing the metrics. Should be in [0, 1] to average.
#'
#' @return A dataframe with a column for each metric, and column 'horizon'.
#'
#' @export
performance_metrics <- function(df, metrics = NULL, rolling_window = 0.1) {
  valid_metrics <- c('mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage')
  if (is.null(metrics)) {
    metrics <- valid_metrics
  }
  if ((!('yhat_lower' %in% colnames(df)) | !('yhat_upper' %in% colnames(df))) & ('coverage' %in% metrics)){
    metrics <- metrics[metrics != 'coverage']
  }

  if (length(metrics) != length(unique(metrics))) {
    stop('Input metrics must be an array of unique values.')
  }
  if (!all(metrics %in% valid_metrics)) {
    stop(
      paste('Valid values for metrics are:', paste(valid_metrics, collapse = ", "))
    )
  }
  df_m <- df
  df_m$horizon <- df_m$ds - df_m$cutoff
  df_m <- df_m[order(df_m$horizon),]
  if (('mape' %in% metrics) & (min(abs(df_m$y)) < 1e-8)) {
    message('Skipping MAPE because y close to 0')
    metrics <- metrics[metrics != 'mape']
  }
  if (('mdape' %in% metrics) & (min(abs(df_m$y)) < 1e-8)) {
    message('Skipping MDAPE because y close to 0')
    metrics <- metrics[metrics != 'mdape']
  }
  if (length(metrics) == 0) {
    return(NULL)
  }
  w <- as.integer(rolling_window * nrow(df_m))
  if (w >= 0) {
    w <- max(w, 1)
    w <- min(w, nrow(df_m))
  }
  # Compute all metrics
  dfs = list()
  for (metric in metrics) {
    dfs[[metric]] <- get(metric)(df_m, w)
  }
  res <- dfs[[metrics[1]]]
  for (i in 2:length(metrics)) {
    res_m <- dfs[[metrics[i]]]
    stopifnot(res$horizon == res_m$horizon)
    res[[metrics[i]]] = res_m[[metrics[i]]]
  }
  return(res)
}

#' Compute a rolling mean of x, after first aggregating by h
#'
#' Right-aligned. Computes a single mean for each unique value of h. Each mean
#' is over at least w samples.
#'
#' @param x Array.
#' @param h Array of horizon for each value in x.
#' @param w Integer window size (number of elements).
#' @param name String name for metric in result dataframe.
#'
#' @return Dataframe with columns horizon and name, the rolling mean of x.
#'
#' @importFrom dplyr "%>%"
#' @keywords internal
rolling_mean_by_h <- function(x, h, w, name) {
  # Aggregate over h
  df <- data.frame(x=x, h=h)
  df2 <- df %>%
    dplyr::group_by(h) %>%
    dplyr::summarise(mean = mean(x), n = dplyr::n())

  xm <- df2$mean
  ns <- df2$n
  hs <- df2$h

  res <- data.frame(horizon=c())
  res[[name]] <- c()
  # Start from the right and work backwards
  i <- length(hs)
  while (i > 0) {
    # Construct a mean of at least w samples
    n <- ns[i]
    xbar <- xm[i]
    j <- i - 1
    while ((n < w) & (j > 0)) {
      # Include points from the previous horizon. All of them if still less
      # than w, otherwise just enough to get to w.
      n2 <- min(w - n, ns[j])
      xbar <- xbar * (n / (n + n2)) + xm[j] * (n2 / (n + n2))
      n <- n + n2
      j <- j - 1
    }
    if (n < w) {
      # Ran out of horizons before enough points.
      break
    }
    res.i <- data.frame(horizon=hs[i])
    res.i[[name]] <- xbar
    res <- rbind(res.i, res)
    i <- i - 1
  }
  return(res)
}


#' Compute a rolling median of x, after first aggregating by h
#'
#' Right-aligned. Computes a single median for each unique value of h. Each median
#' is over at least w samples.
#'
#' For each h where there are fewer than w samples, we take samples from the previous h,
#  moving backwards. (In other words, we ~ assume that the x's are shuffled within each h.)
#'
#' @param x Array.
#' @param h Array of horizon for each value in x.
#' @param w Integer window size (number of elements).
#' @param name String name for metric in result dataframe.
#'
#' @return Dataframe with columns horizon and name, the rolling median of x.
#'
#' @importFrom dplyr "%>%"
rolling_median_by_h <- function(x, h, w, name) {
  # Aggregate over h
  df <- data.frame(x=x, h=h)
  grouped <- df %>% dplyr::group_by(h)
  df2 <- grouped %>%
    dplyr::summarise(size=dplyr::n()) %>%
    dplyr::arrange(h) %>%
    dplyr::select(h, size)

  hs <- df2$h
  res <- data.frame(horizon=c())
  res[[name]] <- c()

  # Start from the right and work backwards
  i <- length(hs)
  while (i > 0) {
    h_i <- hs[i]
    xs <- grouped  %>%
      dplyr::filter(h==h_i)
    xs <- xs$x

    next_idx_to_add = which.max(h==h_i) - 1

    while ((length(xs) < w) & (next_idx_to_add > 0)) {
      # Include points from the previous horizon. All of them if still less
      # than w, otherwise just enough to get to w.
      xs <- c(x[next_idx_to_add], xs)
      next_idx_to_add = next_idx_to_add - 1
    }
    if (length(xs) < w) {
      # Ran out of horizons before enough points.
      break
    }
    res.i <- data.frame(horizon=hs[i])
    res.i[[name]] <- stats::median(xs)
    res <- rbind(res.i, res)
    i <- i - 1
  }
  return(res)
}

# The functions below specify performance metrics for cross-validation results.
# Each takes as input the output of cross_validation, and returns the statistic
# as a dataframe, given a window size for rolling aggregation.

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
  if (w < 0) {
    return(data.frame(horizon = df$horizon, mse = se))
  }
  return(rolling_mean_by_h(x = se, h = df$horizon, w = w, name = 'mse'))
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
  res <- mse(df, w)
  res$mse <- sqrt(res$mse)
  names(res)[names(res) == 'mse'] <- 'rmse'
  return(res)
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
  if (w < 0) {
    return(data.frame(horizon = df$horizon, mae = ae))
  }
  return(rolling_mean_by_h(x = ae, h = df$horizon, w = w, name = 'mae'))
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
  if (w < 0) {
    return(data.frame(horizon = df$horizon, mape = ape))
  }
  return(rolling_mean_by_h(x = ape, h = df$horizon, w = w, name = 'mape'))
}


#' Median absolute percent error
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of median absolute percent errors.
#'
#' @keywords internal
mdape <- function(df, w) {
  ape <- abs((df$y - df$yhat) / df$y)
  if (w < 0) {
    return(data.frame(horizon = df$horizon, mdape = ape))
  }
  return(rolling_median_by_h(x = ape, h = df$horizon, w = w, name = 'mdape'))
}


#' Symmetric mean absolute percentage error
#' based on Chen and Yang (2004) formula
#'
#' @param df Cross-validation results dataframe.
#' @param w Aggregation window size.
#'
#' @return Array of symmetric mean absolute percent errors.
#'
#' @keywords internal
smape <- function(df, w) {
  sape <- abs(df$y - df$yhat) / ((abs(df$y) + abs(df$yhat)) / 2)
  if (w < 0) {
    return(data.frame(horizon = df$horizon, smape = sape))
  }
  return(rolling_mean_by_h(x = sape, h = df$horizon, w = w, name = 'smape'))
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
  if (w < 0) {
    return(data.frame(horizon = df$horizon, coverage = is_covered))
  }
  return(
    rolling_mean_by_h(x = is_covered, h = df$horizon, w = w, name = 'coverage')
  )
}
