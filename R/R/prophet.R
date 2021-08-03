# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c(
  "ds", "y", "cap", ".",
  "component", "dow", "doy", "holiday", "holidays", "holidays_lower", "generated_holidays",
  "holidays_upper", "ix", "lower", "n", "stat", "trend", "row_number", "extra_regressors", "col",
  "trend_lower", "trend_upper", "upper", "value", "weekly", "weekly_lower", "weekly_upper",
  "x", "yearly", "yearly_lower", "yearly_upper", "yhat", "yhat_lower", "yhat_upper",
  "country", "year"
))

#' Prophet forecaster.
#'
#' @param df (optional) Dataframe containing the history. Must have columns ds
#'   (date type) and y, the time series. If growth is logistic, then df must
#'   also have a column cap that specifies the capacity at each ds. If not
#'   provided, then the model object will be instantiated but not fit; use
#'   fit.prophet(m, df) to fit the model.
#' @param growth String 'linear', 'logistic', or 'flat' to specify a linear,
#'   logistic or flat trend.
#' @param changepoints Vector of dates at which to include potential
#'   changepoints. If not specified, potential changepoints are selected
#'   automatically.
#' @param n.changepoints Number of potential changepoints to include. Not used
#'   if input `changepoints` is supplied. If `changepoints` is not supplied,
#'   then n.changepoints potential changepoints are selected uniformly from the
#'   first `changepoint.range` proportion of df$ds.
#' @param changepoint.range Proportion of history in which trend changepoints
#'   will be estimated. Defaults to 0.8 for the first 80%. Not used if
#'   `changepoints` is specified.
#' @param yearly.seasonality Fit yearly seasonality. Can be 'auto', TRUE, FALSE,
#'   or a number of Fourier terms to generate.
#' @param weekly.seasonality Fit weekly seasonality. Can be 'auto', TRUE, FALSE,
#'   or a number of Fourier terms to generate.
#' @param daily.seasonality Fit daily seasonality. Can be 'auto', TRUE, FALSE,
#'   or a number of Fourier terms to generate.
#' @param holidays data frame with columns holiday (character) and ds (date
#'   type)and optionally columns lower_window and upper_window which specify a
#'   range of days around the date to be included as holidays. lower_window=-2
#'   will include 2 days prior to the date as holidays. Also optionally can have
#'   a column prior_scale specifying the prior scale for each holiday.
#' @param seasonality.mode 'additive' (default) or 'multiplicative'.
#' @param seasonality.prior.scale Parameter modulating the strength of the
#'   seasonality model. Larger values allow the model to fit larger seasonal
#'   fluctuations, smaller values dampen the seasonality. Can be specified for
#'   individual seasonalities using add_seasonality.
#' @param holidays.prior.scale Parameter modulating the strength of the holiday
#'   components model, unless overridden in the holidays input.
#' @param changepoint.prior.scale Parameter modulating the flexibility of the
#'   automatic changepoint selection. Large values will allow many changepoints,
#'   small values will allow few changepoints.
#' @param mcmc.samples Integer, if greater than 0, will do full Bayesian
#'   inference with the specified number of MCMC samples. If 0, will do MAP
#'   estimation.
#' @param interval.width Numeric, width of the uncertainty intervals provided
#'   for the forecast. If mcmc.samples=0, this will be only the uncertainty in
#'   the trend using the MAP estimate of the extrapolated generative model. If
#'   mcmc.samples>0, this will be integrated over all model parameters, which
#'   will include uncertainty in seasonality.
#' @param uncertainty.samples Number of simulated draws used to estimate
#'   uncertainty intervals. Settings this value to 0 or False will disable
#'   uncertainty estimation and speed up the calculation.
#' @param backend Whether to use the "rstan" or "cmdstanr" backend to fit the
#'   model. If not provided, uses the R_STAN_BACKEND environment variable.
#' @param fit Boolean, if FALSE the model is initialized but not fit.
#' @param ... Additional arguments, passed to \code{\link{fit.prophet}}
#'
#' @return A prophet model.
#'
#' @examples
#' \dontrun{
#' history <- data.frame(ds = seq(as.Date('2015-01-01'), as.Date('2016-01-01'), by = 'd'),
#'                       y = sin(1:366/200) + rnorm(366)/10)
#' m <- prophet(history)
#' }
#'
#' @export
#' @importFrom dplyr "%>%"
#' @import Rcpp
#' @rawNamespace import(RcppParallel, except = LdFlags)
#' @import rlang
#' @useDynLib prophet, .registration = TRUE
prophet <- function(df = NULL,
                    growth = 'linear',
                    changepoints = NULL,
                    n.changepoints = 25,
                    changepoint.range = 0.8,
                    yearly.seasonality = 'auto',
                    weekly.seasonality = 'auto',
                    daily.seasonality = 'auto',
                    holidays = NULL,
                    seasonality.mode = 'additive',
                    seasonality.prior.scale = 10,
                    holidays.prior.scale = 10,
                    changepoint.prior.scale = 0.05,
                    mcmc.samples = 0,
                    interval.width = 0.80,
                    uncertainty.samples = 1000,
                    fit = TRUE,
                    backend = NULL,
                    ...
) {
  if (!is.null(changepoints)) {
    n.changepoints <- length(changepoints)
  }

  if (is.null(backend)) backend <- get_stan_backend()

  m <- list(
    growth = growth,
    changepoints = changepoints,
    n.changepoints = n.changepoints,
    changepoint.range = changepoint.range,
    yearly.seasonality = yearly.seasonality,
    weekly.seasonality = weekly.seasonality,
    daily.seasonality = daily.seasonality,
    holidays = holidays,
    seasonality.mode = seasonality.mode,
    seasonality.prior.scale = seasonality.prior.scale,
    changepoint.prior.scale = changepoint.prior.scale,
    holidays.prior.scale = holidays.prior.scale,
    mcmc.samples = mcmc.samples,
    interval.width = interval.width,
    uncertainty.samples = uncertainty.samples,
    backend = backend,
    specified.changepoints = !is.null(changepoints),
    start = NULL,  # This and following attributes are set during fitting
    y.scale = NULL,
    logistic.floor = FALSE,
    t.scale = NULL,
    changepoints.t = NULL,
    seasonalities = list(),
    extra_regressors = list(),
    country_holidays = NULL,
    stan.fit = NULL,
    params = list(),
    history = NULL,
    history.dates = NULL,
    train.holiday.names = NULL,
    train.component.cols = NULL,
    component.modes = NULL,
    fit.kwargs = list()
  )
  m <- validate_inputs(m)
  class(m) <- append("prophet", class(m))
  if ((fit) && (!is.null(df))) {
    m <- fit.prophet(m, df, ...)
  }
  return(m)
}

#' Validates the inputs to Prophet.
#'
#' @param m Prophet object.
#'
#' @return The Prophet object.
#'
#' @keywords internal
validate_inputs <- function(m) {
  if (!(m$growth %in% c('linear', 'logistic', 'flat'))) {
    stop("Parameter 'growth' should be 'linear', 'logistic', or 'flat'.")
  }
  if ((m$changepoint.range < 0) | (m$changepoint.range > 1)) {
    stop("Parameter 'changepoint.range' must be in [0, 1]")
  }
  if (!is.null(m$holidays)) {
    if (!(exists('holiday', where = m$holidays))) {
      stop('Holidays dataframe must have holiday field.')
    }
    if (!(exists('ds', where = m$holidays))) {
      stop('Holidays dataframe must have ds field.')
    }
    m$holidays$ds <- as.Date(m$holidays$ds)
    if (any(is.na(m$holidays$ds)) | any(is.na(m$holidays$holiday))) {
      stop('Found NA in the holidays dataframe.')
    }
    has.lower <- exists('lower_window', where = m$holidays)
    has.upper <- exists('upper_window', where = m$holidays)
    if (has.lower + has.upper == 1) {
      stop(paste('Holidays must have both lower_window and upper_window,',
                 'or neither.'))
    }
    if (has.lower) {
      if(max(m$holidays$lower_window, na.rm=TRUE) > 0) {
        stop('Holiday lower_window should be <= 0')
      }
      if(min(m$holidays$upper_window, na.rm=TRUE) < 0) {
        stop('Holiday upper_window should be >= 0')
      }
    }
    for (h in unique(m$holidays$holiday)) {
      validate_column_name(m, h, check_holidays = FALSE)
    }
  }
  if (!(m$seasonality.mode %in% c('additive', 'multiplicative'))) {
    stop("seasonality.mode must be 'additive' or 'multiplicative'")
  }
  return(m)
}

#' Validates the name of a seasonality, holiday, or regressor.
#'
#' @param m Prophet object.
#' @param name string
#' @param check_holidays bool check if name already used for holiday
#' @param check_seasonalities bool check if name already used for seasonality
#' @param check_regressors bool check if name already used for regressor
#'
#' @keywords internal
validate_column_name <- function(
  m, name, check_holidays = TRUE, check_seasonalities = TRUE,
  check_regressors = TRUE
) {
  if (grepl("_delim_", name)) {
    stop('Holiday name cannot contain "_delim_"')
  }
  reserved_names = c(
    'trend', 'additive_terms', 'daily', 'weekly', 'yearly',
    'holidays', 'zeros', 'extra_regressors_additive', 'yhat',
    'extra_regressors_multiplicative', 'multiplicative_terms'
  )
  rn_l = paste(reserved_names,"_lower",sep="")
  rn_u = paste(reserved_names,"_upper",sep="")
  reserved_names = c(reserved_names, rn_l, rn_u,
    c("ds", "y", "cap", "floor", "y_scaled", "cap_scaled"))
  if(name %in% reserved_names){
    stop("Name ", name, " is reserved.")
  }
  if(check_holidays & !is.null(m$holidays) &
     (name %in% unique(m$holidays$holiday))){
    stop("Name ", name, " already used for a holiday.")
  }
  if(check_holidays & !is.null(m$country_holidays)){
    if(name %in% get_holiday_names(m$country_holidays)){
      stop("Name ", name, " is a holiday name in ", m$country_holidays, ".")
    }
  }
  if(check_seasonalities & (!is.null(m$seasonalities[[name]]))){
    stop("Name ", name, " already used for a seasonality.")
  }
  if(check_regressors & (!is.null(m$seasonalities[[name]]))){
    stop("Name ", name, " already used for an added regressor.")
  }
}

#' Convert date vector
#'
#' Convert the date to POSIXct object. Timezones are stripped and replaced
#' with GMT.
#'
#' @param ds Date vector
#'
#' @return vector of POSIXct object converted from date
#'
#' @keywords internal
set_date <- function(ds) {
  if (length(ds) == 0) {
    return(NULL)
  }

  if (is.factor(ds)) {
    ds <- as.character(ds)
  }

  # If a datetime, strip timezone and replace with GMT.
  if (lubridate::is.instant(ds)) {
    ds <- as.POSIXct(lubridate::force_tz(ds, "GMT"), tz = "GMT")
  }
  else {
    # Assume it can be coerced into POSIXct
    if (min(nchar(ds), na.rm=TRUE) < 12) {
        ds <- as.POSIXct(ds, format = "%Y-%m-%d", tz = "GMT")
    } else {
        ds <- as.POSIXct(ds, format = "%Y-%m-%d %H:%M:%S", tz = "GMT")
    }
  }

  attr(ds, "tzone") <- "GMT"
  return(ds)
}

#' Time difference between datetimes
#'
#' Compute time difference of two POSIXct objects
#'
#' @param ds1 POSIXct object
#' @param ds2 POSIXct object
#' @param units string units of difference, e.g. 'days' or 'secs'.
#'
#' @return numeric time difference
#'
#' @keywords internal
time_diff <- function(ds1, ds2, units = "days") {
  return(as.numeric(difftime(ds1, ds2, units = units)))
}

#' Prepare dataframe for fitting or predicting.
#'
#' Adds a time index and scales y. Creates auxiliary columns 't', 't_ix',
#' 'y_scaled', and 'cap_scaled'. These columns are used during both fitting
#' and predicting.
#'
#' @param m Prophet object.
#' @param df Data frame with columns ds, y, and cap if logistic growth. Any
#'  specified additional regressors must also be present.
#' @param initialize_scales Boolean set scaling factors in m from df.
#'
#' @return list with items 'df' and 'm'.
#'
#' @keywords internal
setup_dataframe <- function(m, df, initialize_scales = FALSE) {
  if (exists('y', where=df)) {
    df$y <- as.numeric(df$y)
    if (any(is.infinite(df$y))) {
      stop("Found infinity in column y.")
    }
  }
  df$ds <- set_date(df$ds)
  if (anyNA(df$ds)) {
    stop(paste('Unable to parse date format in column ds. Convert to date ',
               'format (%Y-%m-%d or %Y-%m-%d %H:%M:%S) and check that there',
               'are no NAs.'))
  }
  for (name in names(m$extra_regressors)) {
    if (!(name %in% colnames(df))) {
      stop('Regressor "', name, '" missing from dataframe')
    }
    df[[name]] <- as.numeric(df[[name]])
    if (anyNA(df[[name]])) {
      stop('Found NaN in column ', name)
    }
  }
  for (name in names(m$seasonalities)) {
    condition.name = m$seasonalities[[name]]$condition.name
    if (!is.null(condition.name)) {
      if (!(condition.name %in% colnames(df))) {
        stop('Condition "', name, '" missing from dataframe')
      }
      if(!all(df[[condition.name]] %in% c(FALSE,TRUE))) {
        stop('Found non-boolean in column ', name)
      }
      df[[condition.name]] <- as.logical(df[[condition.name]])
    }
  }

  df <- df %>%
    dplyr::arrange(ds)

  m <- initialize_scales_fn(m, initialize_scales, df)

  if (m$logistic.floor) {
    if (!('floor' %in% colnames(df))) {
      stop("Expected column 'floor'.")
    }
  } else {
    df$floor <- 0
  }

  if (m$growth == 'logistic') {
    if (!(exists('cap', where=df))) {
      stop('Capacities must be supplied for logistic growth.')
    }
    if (any(df$cap <= df$floor)) {
      stop('cap must be greater than floor (which defaults to 0).')
    }
    df <- df %>%
      dplyr::mutate(cap_scaled = (cap - floor) / m$y.scale)
  }

  df$t <- time_diff(df$ds, m$start, "secs") / m$t.scale
  if (exists('y', where=df)) {
    df$y_scaled <- (df$y - df$floor) / m$y.scale
  }

  for (name in names(m$extra_regressors)) {
    props <- m$extra_regressors[[name]]
    df[[name]] <- (df[[name]] - props$mu) / props$std
  }
  return(list("m" = m, "df" = df))
}

#' Initialize model scales.
#'
#' Sets model scaling factors using df.
#'
#' @param m Prophet object.
#' @param initialize_scales Boolean set the scales or not.
#' @param df Dataframe for setting scales.
#'
#' @return Prophet object with scales set.
#'
#' @keywords internal
initialize_scales_fn <- function(m, initialize_scales, df) {
  if (!initialize_scales) {
    return(m)
  }
  if ((m$growth == 'logistic') && ('floor' %in% colnames(df))) {
    m$logistic.floor <- TRUE
    floor <- df$floor
  } else {
    floor <- 0
  }
  m$y.scale <- max(abs(df$y - floor))
  if (m$y.scale == 0) {
    m$y.scale <- 1
  }
  m$start <- min(df$ds)
  m$t.scale <- time_diff(max(df$ds), m$start, "secs")
  for (name in names(m$extra_regressors)) {
    standardize <- m$extra_regressors[[name]]$standardize
    n.vals <- length(unique(df[[name]]))
    if (n.vals < 2) {
      standardize <- FALSE
    }
    if (standardize == 'auto') {
      if (n.vals == 2 && all(sort(unique(df[[name]])) == c(0, 1))) {
        # Don't standardize binary variables
        standardize <- FALSE
      } else {
        standardize <- TRUE
      }
    }
    if (standardize) {
      mu <- mean(df[[name]])
      std <- stats::sd(df[[name]])
      m$extra_regressors[[name]]$mu <- mu
      m$extra_regressors[[name]]$std <- std
    }
  }
  return(m)
}

#' Set changepoints
#'
#' Sets m$changepoints to the dates of changepoints. Either:
#' 1) The changepoints were passed in explicitly.
#'   A) They are empty.
#'   B) They are not empty, and need validation.
#' 2) We are generating a grid of them.
#' 3) The user prefers no changepoints be used.
#'
#' @param m Prophet object.
#'
#' @return m with changepoints set.
#'
#' @keywords internal
set_changepoints <- function(m) {
  if (!is.null(m$changepoints)) {
    if (length(m$changepoints) > 0) {
      m$changepoints <- set_date(m$changepoints)
      if (min(m$changepoints) < min(m$history$ds)
          || max(m$changepoints) > max(m$history$ds)) {
        stop('Changepoints must fall within training data.')
      }
    }
  } else {
    # Place potential changepoints evenly through the first changepoint.range
    # proportion of the history.
    hist.size <- floor(nrow(m$history) * m$changepoint.range)
    if (m$n.changepoints + 1 > hist.size) {
      m$n.changepoints <- hist.size - 1
      message('n.changepoints greater than number of observations. Using ',
              m$n.changepoints)
    }
    if (m$n.changepoints > 0) {
      cp.indexes <- round(seq.int(1, hist.size,
                                  length.out = (m$n.changepoints + 1))[-1])
      m$changepoints <- m$history$ds[cp.indexes]
    } else {
      m$changepoints <- c()
    }
  }
  if (length(m$changepoints) > 0) {
    m$changepoints.t <- sort(
      time_diff(m$changepoints, m$start, "secs")) / m$t.scale
  } else {
    m$changepoints.t <- c(0)  # dummy changepoint
  }
  return(m)
}

#' Provides Fourier series components with the specified frequency and order.
#'
#' @param dates Vector of dates.
#' @param period Number of days of the period.
#' @param series.order Number of components.
#'
#' @return Matrix with seasonality features.
#'
#' @keywords internal
fourier_series <- function(dates, period, series.order) {
  t <- time_diff(dates, set_date('1970-01-01 00:00:00'))
  features <- matrix(0, length(t), 2 * series.order)
  for (i in 1:series.order) {
    x <- as.numeric(2 * i * pi * t / period)
    features[, i * 2 - 1] <- sin(x)
    features[, i * 2] <- cos(x)
  }
  return(features)
}

#' Data frame with seasonality features.
#'
#' @param dates Vector of dates.
#' @param period Number of days of the period.
#' @param series.order Number of components.
#' @param prefix Column name prefix.
#'
#' @return Dataframe with seasonality.
#'
#' @keywords internal
make_seasonality_features <- function(dates, period, series.order, prefix) {
  features <- fourier_series(dates, period, series.order)
  colnames(features) <- paste(prefix, 1:ncol(features), sep = '_delim_')
  return(data.frame(features))
}

#' Construct a dataframe of holiday dates.
#'
#' @param m Prophet object.
#' @param dates Vector with dates used for computing seasonality.
#'
#' @return A dataframe of holiday dates, in holiday dataframe format used in
#'  initialization.
#'
#' @importFrom dplyr "%>%"
#' @keywords internal
construct_holiday_dataframe <- function(m, dates) {
  all.holidays <- data.frame()
  if (!is.null(m$holidays)){
    all.holidays <- m$holidays
  }
  if (!is.null(m$country_holidays)) {
    year.list <- as.numeric(unique(format(dates, "%Y")))
    country.holidays.df <- make_holidays_df(year.list, m$country_holidays)
    all.holidays <- suppressWarnings(dplyr::bind_rows(all.holidays, country.holidays.df))
  }
  # If the model has already been fit with a certain set of holidays,
  # make sure we are using those same ones.
  if (!is.null(m$train.holiday.names)) {
    row.to.keep <- which(all.holidays$holiday %in% m$train.holiday.names)
    all.holidays <- all.holidays[row.to.keep,]
    holidays.to.add <- data.frame(
      holiday=setdiff(m$train.holiday.names, all.holidays$holiday)
    )
    all.holidays <- suppressWarnings(dplyr::bind_rows(all.holidays, holidays.to.add))
  }
  return(all.holidays)
}

#' Construct a matrix of holiday features.
#'
#' @param m Prophet object.
#' @param dates Vector with dates used for computing seasonality.
#' @param holidays Dataframe containing holidays, as returned by
#'  construct_holiday_dataframe.
#'
#' @return A list with entries
#'  holiday.features: dataframe with a column for each holiday.
#'  prior.scales: array of prior scales for each holiday column.
#'  holiday.names: array of names of all holidays.
#'
#' @importFrom dplyr "%>%"
#' @keywords internal
make_holiday_features <- function(m, dates, holidays) {
  # Strip dates to be just days, for joining on holidays
  dates <- set_date(format(dates, "%Y-%m-%d"))
  wide <- holidays %>%
    dplyr::mutate(ds = set_date(ds)) %>%
    dplyr::group_by(holiday, ds) %>%
    dplyr::filter(dplyr::row_number() == 1) %>%
    dplyr::do({
      if (exists('lower_window', where = .) && !is.na(.$lower_window)
          && !is.na(.$upper_window)) {
        offsets <- seq(.$lower_window, .$upper_window)
      } else {
        offsets <- c(0)
      }
      names <- paste(.$holiday, '_delim_', ifelse(offsets < 0, '-', '+'),
                     abs(offsets), sep = '')
      dplyr::tibble(ds = .$ds + offsets * 24 * 3600, holiday = names)
    }) %>%
    dplyr::mutate(x = 1.) %>%
    tidyr::spread(holiday, x, fill = 0)

  holiday.features <- data.frame(ds = set_date(dates)) %>%
    dplyr::left_join(wide, by = 'ds') %>%
    dplyr::select(-ds)
  # Make sure column order is consistent
  holiday.features <- holiday.features %>% dplyr::select(sort(names(.)))
  holiday.features[is.na(holiday.features)] <- 0

  # Prior scales
  if (!('prior_scale' %in% colnames(holidays))) {
    holidays$prior_scale <- m$holidays.prior.scale
  }
  prior.scales.list <- list()
  for (name in unique(holidays$holiday)) {
    df.h <- holidays[holidays$holiday == name, ]
    ps <- unique(df.h$prior_scale)
    if (length(ps) > 1) {
      stop('Holiday ', name, ' does not have a consistent prior scale ',
           'specification')
    }
    if (is.na(ps)) {
      ps <- m$holidays.prior.scale
    }
    if (ps <= 0) {
      stop('Prior scale must be > 0.')
    }
    prior.scales.list[[name]] <- ps
  }

  prior.scales <- c()
  for (name in colnames(holiday.features)) {
    sn <- strsplit(name, '_delim_', fixed = TRUE)[[1]][1]
    prior.scales <- c(prior.scales, prior.scales.list[[sn]])
  }
  holiday.names <- names(prior.scales.list)
  if (is.null(m$train.holiday.names)){
    m$train.holiday.names <- holiday.names
  }
  return(list(m = m,
              holiday.features = holiday.features,
              prior.scales = prior.scales,
              holiday.names = holiday.names))
}

#' Add an additional regressor to be used for fitting and predicting.
#'
#' The dataframe passed to `fit` and `predict` will have a column with the
#' specified name to be used as a regressor. When standardize='auto', the
#' regressor will be standardized unless it is binary. The regression
#' coefficient is given a prior with the specified scale parameter.
#' Decreasing the prior scale will add additional regularization. If no
#' prior scale is provided, holidays.prior.scale will be used.
#' Mode can be specified as either 'additive' or 'multiplicative'. If not
#' specified, m$seasonality.mode will be used. 'additive' means the effect of
#' the regressor will be added to the trend, 'multiplicative' means it will
#' multiply the trend.
#'
#' @param m Prophet object.
#' @param name String name of the regressor
#' @param prior.scale Float scale for the normal prior. If not provided,
#'  holidays.prior.scale will be used.
#' @param standardize Bool, specify whether this regressor will be standardized
#'  prior to fitting. Can be 'auto' (standardize if not binary), True, or
#'  False.
#' @param mode Optional, 'additive' or 'multiplicative'. Defaults to
#'  m$seasonality.mode.
#'
#' @return  The prophet model with the regressor added.
#'
#' @export
add_regressor <- function(
  m, name, prior.scale = NULL, standardize = 'auto', mode = NULL
){
  if (!is.null(m$history)) {
    stop('Regressors must be added prior to model fitting.')
  }
  if (make.names(name, allow_ = TRUE) != name) {
    stop("You have provided a name that is not syntactically valid in R, ", name, ". ",
         "A syntactically valid name consists of letters, numbers and the dot or underline, ",
         "characters and starts with a letter or the dot not followed by a number.")
  }
  validate_column_name(m, name, check_regressors = FALSE)
  if (is.null(prior.scale)) {
    prior.scale <- m$holidays.prior.scale
  }
  if (is.null(mode)) {
    mode <- m$seasonality.mode
  }
  if(prior.scale <= 0) {
    stop("Prior scale must be > 0.")
  }
  if (!(mode %in% c('additive', 'multiplicative'))) {
    stop("mode must be 'additive' or 'multiplicative'")
  }
  m$extra_regressors[[name]] <- list(
    prior.scale = prior.scale,
    standardize = standardize,
    mu = 0,
    std = 1.0,
    mode = mode
  )
  return(m)
}

#' Add a seasonal component with specified period, number of Fourier
#' components, and prior scale.
#'
#' Increasing the number of Fourier components allows the seasonality to change
#' more quickly (at risk of overfitting). Default values for yearly and weekly
#' seasonalities are 10 and 3 respectively.
#'
#' Increasing prior scale will allow this seasonality component more
#' flexibility, decreasing will dampen it. If not provided, will use the
#' seasonality.prior.scale provided on Prophet initialization (defaults to 10).
#'
#' Mode can be specified as either 'additive' or 'multiplicative'. If not
#' specified, m$seasonality.mode will be used (defaults to 'additive').
#' Additive means the seasonality will be added to the trend, multiplicative
#' means it will multiply the trend.
#'
#' If condition.name is provided, the dataframe passed to `fit` and `predict`
#' should have a column with the specified condition.name containing booleans
#' which decides when to apply seasonality.
#'
#' @param m Prophet object.
#' @param name String name of the seasonality component.
#' @param period Float number of days in one period.
#' @param fourier.order Int number of Fourier components to use.
#' @param prior.scale Optional float prior scale for this component.
#' @param mode Optional 'additive' or 'multiplicative'.
#' @param condition.name String name of the seasonality condition.
#'
#' @return The prophet model with the seasonality added.
#'
#' @export
add_seasonality <- function(
  m, name, period, fourier.order, prior.scale = NULL, mode = NULL,
  condition.name = NULL
) {
  if (!is.null(m$history)) {
    stop("Seasonality must be added prior to model fitting.")
  }
  if (!(name %in% c('daily', 'weekly', 'yearly'))) {
    # Allow overriding built-in seasonalities
    if (make.names(name, allow_ = TRUE) != name) {
      stop("You have provided a name that is not syntactically valid in R, ", name, ". ",
           "A syntactically valid name consists of letters, numbers and the dot or underline, ",
           "characters and starts with a letter or the dot not followed by a number.")
    }
    validate_column_name(m, name, check_seasonalities = FALSE)
  }
  if (is.null(prior.scale)) {
    ps <- m$seasonality.prior.scale
  } else {
    ps <- prior.scale
  }
  if (ps <= 0) {
    stop('Prior scale must be > 0.')
  }
  if (fourier.order <= 0) {
    stop('Fourier order must be > 0.')
  }
  if (is.null(mode)) {
    mode <- m$seasonality.mode
  }
  if (!(mode %in% c('additive', 'multiplicative'))) {
    stop("mode must be 'additive' or 'multiplicative'")
  }
  if (!is.null(condition.name)) {
    validate_column_name(m, condition.name)
  }
  m$seasonalities[[name]] <- list(
    period = period,
    fourier.order = fourier.order,
    prior.scale = ps,
    mode = mode,
    condition.name = condition.name
  )
  return(m)
}

#' Add in built-in holidays for the specified country.
#'
#' These holidays will be included in addition to any specified on model
#' initialization.
#'
#' Holidays will be calculated for arbitrary date ranges in the history
#' and future. See the online documentation for the list of countries with
#' built-in holidays.
#'
#' Built-in country holidays can only be set for a single country.
#'
#' @param m Prophet object.
#' @param country_name Name of the country, like 'UnitedStates' or 'US'
#'
#' @return The prophet model with the holidays country set.
#'
#' @export
add_country_holidays <- function(m, country_name) {
  if (!is.null(m$history)) {
    stop("Country holidays must be added prior to model fitting.")
  }
  if (!(country_name %in% generated_holidays$country)){
      stop("Holidays in ", country_name," are not currently supported!")
    }
  # Validate names.
  for (name in get_holiday_names(country_name)) {
    # Allow merging with existing holidays
    validate_column_name(m, name, check_holidays = FALSE)
  }
  # Set the holidays.
  if (!is.null(m$country_holidays)) {
    message(
      'Changing country holidays from ', m$country_holidays, ' to ',
      country_name
    )
  }
  m$country_holidays = country_name
  return(m)
}

#' Dataframe with seasonality features.
#' Includes seasonality features, holiday features, and added regressors.
#'
#' @param m Prophet object.
#' @param df Dataframe with dates for computing seasonality features and any
#'  added regressors.
#'
#' @return List with items
#'  seasonal.features: Dataframe with regressor features,
#'  prior.scales: Array of prior scales for each column of the features
#'    dataframe.
#'  component.cols: Dataframe with indicators for which regression components
#'    correspond to which columns.
#'  modes: List with keys 'additive' and 'multiplicative' with arrays of
#'    component names for each mode of seasonality.
#'
#' @keywords internal
make_all_seasonality_features <- function(m, df) {
  seasonal.features <- data.frame(row.names = 1:nrow(df))
  prior.scales <- c()
  modes <- list(additive = c(), multiplicative = c())

  # Seasonality features
  for (name in names(m$seasonalities)) {
    props <- m$seasonalities[[name]]
    features <- make_seasonality_features(
      df$ds, props$period, props$fourier.order, name)
    if (!is.null(props$condition.name)) {
      features[!df[[props$condition.name]],] <- 0
    }
    seasonal.features <- cbind(seasonal.features, features)
    prior.scales <- c(prior.scales,
                      props$prior.scale * rep(1, ncol(features)))
    modes[[props$mode]] <- c(modes[[props$mode]], name)
  }

  # Holiday features
  holidays <- construct_holiday_dataframe(m, df$ds)
  if (nrow(holidays) > 0) {
    out <- make_holiday_features(m, df$ds, holidays)
    m <- out$m
    seasonal.features <- cbind(seasonal.features, out$holiday.features)
    prior.scales <- c(prior.scales, out$prior.scales)
    modes[[m$seasonality.mode]] <- c(
      modes[[m$seasonality.mode]], out$holiday.names
    )
  }

  # Additional regressors
  for (name in names(m$extra_regressors)) {
    props <- m$extra_regressors[[name]]
    seasonal.features[[name]] <- df[[name]]
    prior.scales <- c(prior.scales, props$prior.scale)
    modes[[props$mode]] <- c(modes[[props$mode]], name)
  }

  # Dummy to prevent empty X
  if (ncol(seasonal.features) == 0) {
    seasonal.features <- data.frame(zeros = rep(0, nrow(df)))
    prior.scales <- c(1.)
  }

  components.list <- regressor_column_matrix(m, seasonal.features, modes)
  return(list(m = m,
              seasonal.features = seasonal.features,
              prior.scales = prior.scales,
              component.cols = components.list$component.cols,
              modes = components.list$modes))
}

#' Dataframe indicating which columns of the feature matrix correspond to
#' which seasonality/regressor components.
#'
#' Includes combination components, like 'additive_terms'. These combination
#' components will be added to the 'modes' input.
#'
#' @param m Prophet object.
#' @param seasonal.features Constructed seasonal features dataframe.
#' @param modes List with keys 'additive' and 'multiplicative' with arrays of
#'  component names for each mode of seasonality.
#'
#' @return List with items
#'  component.cols: A binary indicator dataframe with columns seasonal
#'    components and rows columns in seasonal.features. Entry is 1 if that
#'    column is used in that component.
#'  modes: Updated input with combination components.
#'
#' @keywords internal
regressor_column_matrix <- function(m, seasonal.features, modes) {
  components <- dplyr::tibble(component = colnames(seasonal.features)) %>%
    dplyr::mutate(col = seq_len(dplyr::n())) %>%
    tidyr::separate(component, c('component', 'part'), sep = "_delim_",
                    extra = "merge", fill = "right") %>%
    dplyr::select(col, component)
  # Add total for holidays
  if(!is.null(m$train.holiday.names)){
    components <- add_group_component(
      components, 'holidays', unique(m$train.holiday.names))
  }
  # Add totals for additive and multiplicative components, and regressors
  for (mode in c('additive', 'multiplicative')) {
    components <- add_group_component(
      components, paste0(mode, '_terms'), modes[[mode]])
    regressors_by_mode <- c()
    for (name in names(m$extra_regressors)) {
      if (m$extra_regressors[[name]]$mode == mode) {
        regressors_by_mode <- c(regressors_by_mode, name)
      }
    }
    components <- add_group_component(
      components, paste0('extra_regressors_', mode), regressors_by_mode)
    # Add combination components to modes
    modes[[mode]] <- c(modes[[mode]], paste0(mode, '_terms'))
    modes[[mode]] <- c(modes[[mode]], paste0('extra_regressors_', mode))
  }
  # After all of the additive/multiplicative groups have been added,
  modes[[m$seasonality.mode]] <- c(modes[[m$seasonality.mode]], 'holidays')
  # Convert to a binary matrix
  component.cols <- as.data.frame.matrix(
    table(components$col, components$component)
  )
  component.cols <- (
    component.cols[order(as.numeric(row.names(component.cols))), ,
                   drop = FALSE]
  )
  # Add columns for additive and multiplicative terms, if missing
  for (name in c('additive_terms', 'multiplicative_terms')) {
    if (!(name %in% colnames(component.cols))) {
      component.cols[[name]] <- 0
    }
  }
  # Remove the placeholder
  components <- dplyr::filter(components, component != 'zeros')
  # Validation
  if (
    max(component.cols$additive_terms
    + component.cols$multiplicative_terms) > 1
  ) {
    stop('A bug occurred in seasonal components.')
  }
  # Compare to training, if set.
  if (!is.null(m$train.component.cols)) {
    component.cols <- component.cols[, colnames(m$train.component.cols)]
    if (!all(component.cols == m$train.component.cols)) {
      stop('A bug occurred in constructing regressors.')
    }
  }
  return(list(component.cols = component.cols, modes = modes))
}

#' Adds a component with given name that contains all of the components
#' in group.
#'
#' @param components Dataframe with components.
#' @param name Name of new group component.
#' @param group  List of components that form the group.
#'
#' @return Dataframe with components.
#'
#' @keywords internal
add_group_component <- function(components, name, group) {
  new_comp <- components[(components$component %in% group), ]
  group_cols <- unique(new_comp$col)
  if (length(group_cols) > 0) {
    new_comp <- data.frame(col=group_cols, component=name)
    components <- rbind(components, new_comp)
  }
  return(components)
}

#' Get number of Fourier components for built-in seasonalities.
#'
#' @param m Prophet object.
#' @param name String name of the seasonality component.
#' @param arg 'auto', TRUE, FALSE, or number of Fourier components as
#'  provided.
#' @param auto.disable Bool if seasonality should be disabled when 'auto'.
#' @param default.order Int default Fourier order.
#'
#' @return Number of Fourier components, or 0 for disabled.
#'
#' @keywords internal
parse_seasonality_args <- function(m, name, arg, auto.disable, default.order) {
  if (arg == 'auto') {
    fourier.order <- 0
    if (name %in% names(m$seasonalities)) {
      message('Found custom seasonality named "', name,
              '", disabling built-in ', name, ' seasonality.')
    } else if (auto.disable) {
      message('Disabling ', name, ' seasonality. Run prophet with ', name,
              '.seasonality=TRUE to override this.')
    } else {
      fourier.order <- default.order
    }
  } else if (arg == TRUE) {
    fourier.order <- default.order
  } else if (arg == FALSE) {
    fourier.order <- 0
  } else {
    fourier.order <- arg
  }
  return(fourier.order)
}

#' Set seasonalities that were left on auto.
#'
#' Turns on yearly seasonality if there is >=2 years of history.
#' Turns on weekly seasonality if there is >=2 weeks of history, and the
#' spacing between dates in the history is <7 days.
#' Turns on daily seasonality if there is >=2 days of history, and the spacing
#' between dates in the history is <1 day.
#'
#' @param m Prophet object.
#'
#' @return The prophet model with seasonalities set.
#'
#' @keywords internal
set_auto_seasonalities <- function(m) {
  first <- min(m$history$ds)
  last <- max(m$history$ds)
  dt <- diff(time_diff(m$history$ds, m$start))
  min.dt <- min(dt[dt > 0])

  yearly.disable <- time_diff(last, first) < 730
  fourier.order <- parse_seasonality_args(
    m, 'yearly', m$yearly.seasonality, yearly.disable, 10)
  if (fourier.order > 0) {
    m$seasonalities[['yearly']] <- list(
      period = 365.25,
      fourier.order = fourier.order,
      prior.scale = m$seasonality.prior.scale,
      mode = m$seasonality.mode,
      condition.name = NULL
    )
  }

  weekly.disable <- ((time_diff(last, first) < 14) || (min.dt >= 7))
  fourier.order <- parse_seasonality_args(
    m, 'weekly', m$weekly.seasonality, weekly.disable, 3)
  if (fourier.order > 0) {
    m$seasonalities[['weekly']] <- list(
      period = 7,
      fourier.order = fourier.order,
      prior.scale = m$seasonality.prior.scale,
      mode = m$seasonality.mode,
      condition.name = NULL
    )
  }

  daily.disable <- ((time_diff(last, first) < 2) || (min.dt >= 1))
  fourier.order <- parse_seasonality_args(
    m, 'daily', m$daily.seasonality, daily.disable, 4)
  if (fourier.order > 0) {
    m$seasonalities[['daily']] <- list(
      period = 1,
      fourier.order = fourier.order,
      prior.scale = m$seasonality.prior.scale,
      mode = m$seasonality.mode,
      condition.name = NULL
    )
  }
  return(m)
}

#' Initialize flat growth.
#'
#' Provides a strong initialization for flat growth by setting the
#' growth to 0 and calculates the offset parameter that pass the
#' function through the mean of the the y_scaled values.
#'
#' @param df Data frame with columns ds (date), y_scaled (scaled time series),
#'  and t (scaled time).
#'
#' @return A vector (k, m) with the rate (k) and offset (m) of the flat
#'  growth function.
#'
#' @keywords internal
flat_growth_init <- function(df) {
  # Initialize the rate
  k <- 0
  # And the offset
  m <- mean(df$y_scaled)
  return(c(k, m))
}

#' Initialize constant growth.
#'
#' Provides a strong initialization for linear growth by calculating the
#' growth and offset parameters that pass the function through the first and
#' last points in the time series.
#'
#' @param df Data frame with columns ds (date), y_scaled (scaled time series),
#'  and t (scaled time).
#'
#' @return A vector (k, m) with the rate (k) and offset (m) of the linear
#'  growth function.
#'
#' @keywords internal
linear_growth_init <- function(df) {
  i0 <- which.min(df$ds)
  i1 <- which.max(df$ds)
  T <- df$t[i1] - df$t[i0]
  # Initialize the rate
  k <- (df$y_scaled[i1] - df$y_scaled[i0]) / T
  # And the offset
  m <- df$y_scaled[i0] - k * df$t[i0]
  return(c(k, m))
}

#' Initialize logistic growth.
#'
#' Provides a strong initialization for logistic growth by calculating the
#' growth and offset parameters that pass the function through the first and
#' last points in the time series.
#'
#' @param df Data frame with columns ds (date), cap_scaled (scaled capacity),
#'  y_scaled (scaled time series), and t (scaled time).
#'
#' @return A vector (k, m) with the rate (k) and offset (m) of the logistic
#'  growth function.
#'
#' @keywords internal
logistic_growth_init <- function(df) {
  i0 <- which.min(df$ds)
  i1 <- which.max(df$ds)
  T <- df$t[i1] - df$t[i0]

  # Force valid values, in case y > cap or y < 0
  C0 <- df$cap_scaled[i0]
  C1 <- df$cap_scaled[i1]
  y0 <- max(0.01 * C0, min(0.99 * C0, df$y_scaled[i0]))
  y1 <- max(0.01 * C1, min(0.99 * C1, df$y_scaled[i1]))

  r0 <- C0 / y0
  r1 <- C1 / y1

  if (abs(r0 - r1) <= 0.01) {
    r0 <- 1.05 * r0
  }
  L0 <- log(r0 - 1)
  L1 <- log(r1 - 1)
  # Initialize the offset
  m <- L0 * T / (L0 - L1)
  # And the rate
  k <- (L0 - L1) / T
  return(c(k, m))
}

#' Fit the prophet model.
#'
#' This sets m$params to contain the fitted model parameters. It is a list
#' with the following elements:
#'   k (M array): M posterior samples of the initial slope.
#'   m (M array): The initial intercept.
#'   delta (MxN matrix): The slope change at each of N changepoints.
#'   beta (MxK matrix): Coefficients for K seasonality features.
#'   sigma_obs (M array): Noise level.
#' Note that M=1 if MAP estimation.
#'
#' @param m Prophet object.
#' @param df Data frame.
#' @param ... Additional arguments passed to the \code{optimizing} or
#'  \code{sampling} functions in Stan.
#'
#' @export
fit.prophet <- function(m, df, ...) {
  if (!is.null(m$history)) {
    stop("Prophet object can only be fit once. Instantiate a new object.")
  }
  if (!(exists('ds', where = df)) | !(exists('y', where = df))) {
    stop(paste(
      "Dataframe must have columns 'ds' and 'y' with the dates and values",
      "respectively."
    ))
  }
  history <- df %>%
    dplyr::filter(!is.na(y))
  if (nrow(history) < 2) {
    stop("Dataframe has less than 2 non-NA rows.")
  }
  m$history.dates <- sort(set_date(unique(df$ds)))

  out <- setup_dataframe(m, history, initialize_scales = TRUE)
  history <- out$df
  m <- out$m
  m$history <- history
  m <- set_auto_seasonalities(m)
  out2 <- make_all_seasonality_features(m, history)
  m <- out2$m
  seasonal.features <- out2$seasonal.features
  prior.scales <- out2$prior.scales
  component.cols <- out2$component.cols
  m$train.component.cols <- component.cols
  m$component.modes <- out2$modes
  m$fit.kwargs <- list(...)

  m <- set_changepoints(m)

  # Construct input to stan
  dat <- list(
    T = nrow(history),
    K = ncol(seasonal.features),
    S = length(m$changepoints.t),
    y = history$y_scaled,
    t = history$t,
    t_change = array(m$changepoints.t),
    X = as.matrix(seasonal.features),
    sigmas = array(prior.scales),
    tau = m$changepoint.prior.scale,
    trend_indicator = switch(m$growth, 'linear'=0, 'logistic'=1, 'flat'=2),
    s_a = array(component.cols$additive_terms),
    s_m = array(component.cols$multiplicative_terms)
  )

  # Run stan
  if (m$growth == 'linear') {
    dat$cap <- rep(0, nrow(history))  # Unused inside Stan
    kinit <- linear_growth_init(history)
  } else if (m$growth == 'flat') {
    dat$cap <- rep(0, nrow(history)) # Unused inside Stan
    kinit <- flat_growth_init(history)
  } else if (m$growth == 'logistic') {
    dat$cap <- history$cap_scaled  # Add capacities to the Stan data
    kinit <- logistic_growth_init(history)
  }

  model <- .load_model(m$backend)

  stan_init <- function() {
    list(k = kinit[1],
         m = kinit[2],
         delta = array(rep(0, length(m$changepoints.t))),
         beta = array(rep(0, ncol(seasonal.features))),
         sigma_obs = 1
    )
  }

  if (min(history$y) == max(history$y) &
        (m$growth %in% c('linear', 'flat'))) {
    # Nothing to fit.
    m$params <- stan_init()
    m$params$sigma_obs <- 0.
    n.iteration <- 1.
  } else {
    if (m$mcmc.samples > 0) {
      args <- .stan_args(model, dat, stan_init, m$backend, type = "mcmc", m$mcmc.samples, ...)
      model_output <- .sampling(args, m$backend)
    } else {
      args <- .stan_args(model, dat, stan_init, m$backend, type = "optimize", ...)
      model_output <- .fit(args, m$backend)
    }
    m$stan.fit <- model_output$stan_fit
    m$params <- model_output$params
    n.iteration <- model_output$n_iteration
  }
  # Cast the parameters to have consistent form, whether full bayes or MAP
  for (name in c('delta', 'beta')){
    m$params[[name]] <- matrix(m$params[[name]], nrow = n.iteration)
  }
  # rstan::sampling returns 1d arrays; converts to atomic vectors.
  for (name in c('k', 'm', 'sigma_obs')){
    m$params[[name]] <- c(m$params[[name]])
  }
  # If no changepoints were requested, replace delta with 0s
  if (m$n.changepoints == 0) {
    # Fold delta into the base rate k
    m$params$k <- m$params$k + m$params$delta[, 1]
    m$params$delta <- matrix(rep(0, length(m$params$delta)), nrow = n.iteration)
  }
  return(m)
}

#' Predict using the prophet model.
#'
#' @param object Prophet object.
#' @param df Dataframe with dates for predictions (column ds), and capacity
#'  (column cap) if logistic growth. If not provided, predictions are made on
#'  the history.
#' @param ... additional arguments.
#'
#' @return A dataframe with the forecast components.
#'
#' @examples
#' \dontrun{
#' history <- data.frame(ds = seq(as.Date('2015-01-01'), as.Date('2016-01-01'), by = 'd'),
#'                       y = sin(1:366/200) + rnorm(366)/10)
#' m <- prophet(history)
#' future <- make_future_dataframe(m, periods = 365)
#' forecast <- predict(m, future)
#' plot(m, forecast)
#' }
#'
#' @export
predict.prophet <- function(object, df = NULL, ...) {
  if (is.null(object$history)) {
    stop("Model must be fit before predictions can be made.")
  }
  if (is.null(df)) {
    df <- object$history
  } else {
    if (nrow(df) == 0) {
      stop("Dataframe has no rows.")
    }
    out <- setup_dataframe(object, df)
    df <- out$df
  }

  df$trend <- predict_trend(object, df)
  seasonal.components <- predict_seasonal_components(object, df)
  if (object$uncertainty.samples) {
    intervals <- predict_uncertainty(object, df)
  } else {
    intervals <- NULL
    }

  # Drop columns except ds, cap, floor, and trend
  cols <- c('ds', 'trend')
  if ('cap' %in% colnames(df)) {
    cols <- c(cols, 'cap')
  }
  if (object$logistic.floor) {
    cols <- c(cols, 'floor')
  }
  df <- df[cols]
  df <- dplyr::bind_cols(df, seasonal.components, intervals)
  df$yhat <- df$trend * (1 + df$multiplicative_terms) + df$additive_terms
  return(df)
}

#' Evaluate the flat trend function.
#'
#' @param t Vector of times on which the function is evaluated.
#' @param m Float initial offset.
#'
#' @return Vector y(t).
#'
#' @keywords internal
flat_trend <- function(t, m) {
  y <- rep(m, length(t))
  return(y)
}

#' Evaluate the piecewise linear function.
#'
#' @param t Vector of times on which the function is evaluated.
#' @param deltas Vector of rate changes at each changepoint.
#' @param k Float initial rate.
#' @param m Float initial offset.
#' @param changepoint.ts Vector of changepoint times.
#'
#' @return Vector y(t).
#'
#' @keywords internal
piecewise_linear <- function(t, deltas, k, m, changepoint.ts) {
  # Intercept changes
  gammas <- -changepoint.ts * deltas
  # Get cumulative slope and intercept at each t
  k_t <- rep(k, length(t))
  m_t <- rep(m, length(t))
  for (s in 1:length(changepoint.ts)) {
    indx <- t >= changepoint.ts[s]
    k_t[indx] <- k_t[indx] + deltas[s]
    m_t[indx] <- m_t[indx] + gammas[s]
  }
  y <- k_t * t + m_t
  return(y)
}

#' Evaluate the piecewise logistic function.
#'
#' @param t Vector of times on which the function is evaluated.
#' @param cap Vector of capacities at each t.
#' @param deltas Vector of rate changes at each changepoint.
#' @param k Float initial rate.
#' @param m Float initial offset.
#' @param changepoint.ts Vector of changepoint times.
#'
#' @return Vector y(t).
#'
#' @keywords internal
piecewise_logistic <- function(t, cap, deltas, k, m, changepoint.ts) {
  # Compute offset changes
  k.cum <- c(k, cumsum(deltas) + k)
  gammas <- rep(0, length(changepoint.ts))
  for (i in 1:length(changepoint.ts)) {
    gammas[i] <- ((changepoint.ts[i] - m - sum(gammas))
                  * (1 - k.cum[i] / k.cum[i + 1]))
  }
  # Get cumulative rate and offset at each t
  k_t <- rep(k, length(t))
  m_t <- rep(m, length(t))
  for (s in 1:length(changepoint.ts)) {
    indx <- t >= changepoint.ts[s]
    k_t[indx] <- k_t[indx] + deltas[s]
    m_t[indx] <- m_t[indx] + gammas[s]
  }
  y <- cap / (1 + exp(-k_t * (t - m_t)))
  return(y)
}

#' Predict trend using the prophet model.
#'
#' @param model Prophet object.
#' @param df Prediction dataframe.
#'
#' @return Vector with trend on prediction dates.
#'
#' @keywords internal
predict_trend <- function(model, df) {
  k <- mean(model$params$k, na.rm = TRUE)
  param.m <- mean(model$params$m, na.rm = TRUE)
  deltas <- colMeans(model$params$delta, na.rm = TRUE)

  t <- df$t
  if (model$growth == 'linear') {
    trend <- piecewise_linear(t, deltas, k, param.m, model$changepoints.t)
  } else if (model$growth == 'flat') {
     trend <- flat_trend(t, param.m)
  } else if (model$growth == 'logistic') {
    cap <- df$cap_scaled
    trend <- piecewise_logistic(
      t, cap, deltas, k, param.m, model$changepoints.t)
  }
  return(trend * model$y.scale + df$floor)
}

#' Predict seasonality components, holidays, and added regressors.
#'
#' @param m Prophet object.
#' @param df Prediction dataframe.
#'
#' @return Dataframe with seasonal components.
#'
#' @keywords internal
predict_seasonal_components <- function(m, df) {
  out <- make_all_seasonality_features(m, df)
  m <- out$m
  seasonal.features <- out$seasonal.features
  component.cols <- out$component.cols
  if (m$uncertainty.samples){
    lower.p <- (1 - m$interval.width)/2
    upper.p <- (1 + m$interval.width)/2
  }

  X <- as.matrix(seasonal.features)
  component.predictions <- data.frame(matrix(ncol = 0, nrow = nrow(X)))
  for (component in colnames(component.cols)) {
    beta.c <- t(m$params$beta) * component.cols[[component]]

    comp <- X %*% beta.c
    if (component %in% m$component.modes$additive) {
      comp <- comp * m$y.scale
    }
    component.predictions[[component]] <- rowMeans(comp, na.rm = TRUE)
    if (m$uncertainty.samples){
      component.predictions[[paste0(component, '_lower')]] <- apply(
        comp, 1, stats::quantile, lower.p, na.rm = TRUE)
      component.predictions[[paste0(component, '_upper')]] <- apply(
        comp, 1, stats::quantile, upper.p, na.rm = TRUE)
    }
  }
  return(component.predictions)
}

#' Prophet posterior predictive samples.
#'
#' @param m Prophet object.
#' @param df Prediction dataframe.
#'
#' @return List with posterior predictive samples for the forecast yhat and
#'  for the trend component.
#'
#' @keywords internal
sample_posterior_predictive <- function(m, df) {
  # Sample trend, seasonality, and yhat from the extrapolation model.
  n.iterations <- length(m$params$k)
  samp.per.iter <- max(1, ceiling(m$uncertainty.samples / n.iterations))
  nsamp <- n.iterations * samp.per.iter  # The actual number of samples

  out <- make_all_seasonality_features(m, df)
  seasonal.features <- out$seasonal.features
  component.cols <- out$component.cols
  sim.values <- list("trend" = matrix(, nrow = nrow(df), ncol = nsamp),
                     "yhat" = matrix(, nrow = nrow(df), ncol = nsamp))

  for (i in 1:n.iterations) {
    # For each set of parameters from MCMC (or just 1 set for MAP),
    for (j in 1:samp.per.iter) {
      # Do a simulation with this set of parameters,
      sim <- sample_model(
        m = m,
        df = df,
        seasonal.features = seasonal.features,
        iteration = i,
        s_a = component.cols$additive_terms,
        s_m = component.cols$multiplicative_terms
      )
      # Store the results
      for (key in c("trend", "yhat")) {
        sim.values[[key]][,(i - 1) * samp.per.iter + j] <- sim[[key]]
      }
    }
  }
  return(sim.values)
}

#' Sample from the posterior predictive distribution.
#'
#' @param m Prophet object.
#' @param df Dataframe with dates for predictions (column ds), and capacity
#'  (column cap) if logistic growth.
#'
#' @return A list with items "trend" and "yhat" containing
#'  posterior predictive samples for that component.
#'
#' @export
predictive_samples <- function(m, df) {
  df <- setup_dataframe(m, df)$df
  sim.values <- sample_posterior_predictive(m, df)
  return(sim.values)
}

#' Prophet uncertainty intervals for yhat and trend
#'
#' @param m Prophet object.
#' @param df Prediction dataframe.
#'
#' @return Dataframe with uncertainty intervals.
#'
#' @keywords internal
predict_uncertainty <- function(m, df) {
  sim.values <- sample_posterior_predictive(m, df)
  # Add uncertainty estimates
  lower.p <- (1 - m$interval.width)/2
  upper.p <- (1 + m$interval.width)/2

  intervals <- cbind(
    t(apply(t(sim.values$yhat), 2, stats::quantile, c(lower.p, upper.p),
            na.rm = TRUE)),
    t(apply(t(sim.values$trend), 2, stats::quantile, c(lower.p, upper.p),
            na.rm = TRUE))
  )

  colnames(intervals) <- paste(rep(c('yhat', 'trend'), each=2),
                               c('lower', 'upper'), sep = "_")

  return(dplyr::as_tibble(intervals))
}

#' Simulate observations from the extrapolated generative model.
#'
#' @param m Prophet object.
#' @param df Prediction dataframe.
#' @param seasonal.features Data frame of seasonal features
#' @param iteration Int sampling iteration to use parameters from.
#' @param s_a Indicator vector for additive components
#' @param s_m Indicator vector for multiplicative components
#'
#' @return List of trend and yhat, each a vector like df$t.
#'
#' @keywords internal
sample_model <- function(m, df, seasonal.features, iteration, s_a, s_m) {
  trend <- sample_predictive_trend(m, df, iteration)

  beta <- m$params$beta[iteration,]
  Xb_a = as.matrix(seasonal.features) %*% (beta * s_a) * m$y.scale
  Xb_m = as.matrix(seasonal.features) %*% (beta * s_m)

  sigma <- m$params$sigma_obs[iteration]
  noise <- stats::rnorm(nrow(df), mean = 0, sd = sigma) * m$y.scale

  return(list("yhat" = trend * (1 + Xb_m) + Xb_a + noise,
              "trend" = trend))
}

#' Simulate the trend using the extrapolated generative model.
#'
#' @param model Prophet object.
#' @param df Prediction dataframe.
#' @param iteration Int sampling iteration to use parameters from.
#'
#' @return Vector of simulated trend over df$t.
#'
#' @keywords internal
sample_predictive_trend <- function(model, df, iteration) {
  k <- model$params$k[iteration]
  param.m <- model$params$m[iteration]
  deltas <- model$params$delta[iteration,]

  t <- df$t
  T <- max(t)

  # New changepoints from a Poisson process with rate S on [1, T]
  if (T > 1) {
    S <- length(model$changepoints.t)
    n.changes <- stats::rpois(1, S * (T - 1))
  } else {
    n.changes <- 0
  }
  if (n.changes > 0) {
    changepoint.ts.new <- 1 + stats::runif(n.changes) * (T - 1)
    changepoint.ts.new <- sort(changepoint.ts.new)
  } else {
    changepoint.ts.new <- c()
  }

  # Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
  lambda <- mean(abs(c(deltas))) + 1e-8
  # Sample deltas
  deltas.new <- extraDistr::rlaplace(n.changes, mu = 0, sigma = lambda)

  # Combine with changepoints from the history
  changepoint.ts <- c(model$changepoints.t, changepoint.ts.new)
  deltas <- c(deltas, deltas.new)

  # Get the corresponding trend
  if (model$growth == 'linear') {
    trend <- piecewise_linear(t, deltas, k, param.m, changepoint.ts)
  } else if (model$growth == 'flat') {
    trend <- flat_trend(t, param.m)
  } else if (model$growth == 'logistic') {
    cap <- df$cap_scaled
    trend <- piecewise_logistic(t, cap, deltas, k, param.m, changepoint.ts)
  }
  return(trend * model$y.scale + df$floor)
}

#' Make dataframe with future dates for forecasting.
#'
#' @param m Prophet model object.
#' @param periods Int number of periods to forecast forward.
#' @param freq 'day', 'week', 'month', 'quarter', 'year', 1(1 sec), 60(1 minute) or 3600(1 hour).
#' @param include_history Boolean to include the historical dates in the data
#'  frame for predictions.
#'
#' @return Dataframe that extends forward from the end of m$history for the
#'  requested number of periods.
#'
#' @export
make_future_dataframe <- function(m, periods, freq = 'day',
                                  include_history = TRUE) {
  # For backwards compatibility with previous zoo date type,
  if (freq == 'm') {
    freq <- 'month'
  }
  if (is.null(m$history.dates)) {
    stop('Model must be fit before this can be used.')
  }
  dates <- seq(max(m$history.dates), length.out = periods + 1, by = freq)
  dates <- dates[2:(periods + 1)]  # Drop the first, which is max(history$ds)
  if (include_history) {
    dates <- c(m$history.dates, dates)
    attr(dates, "tzone") <- "GMT"
  }
  return(data.frame(ds = dates))
}
