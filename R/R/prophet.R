## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.

## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant 
## of patent rights can be found in the PATENTS file in the same directory.

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c(
  "ds", "y", "cap", ".",
  "component", "dow", "doy", "holiday", "holidays", "holidays_lower", "holidays_upper", "ix",
  "lower", "n", "stat", "trend", "row_number",
  "trend_lower", "trend_upper", "upper", "value", "weekly", "weekly_lower", "weekly_upper",
  "x", "yearly", "yearly_lower", "yearly_upper", "yhat", "yhat_lower", "yhat_upper"))

#' Prophet forecaster.
#'
#' @param df Data frame with columns ds (date type) and y, the time series.
#'  If growth is logistic, then df must also have a column cap that specifies
#'  the capacity at each ds.
#' @param growth String 'linear' or 'logistic' to specify a linear or logistic
#'  trend.
#' @param changepoints Vector of dates at which to include potential
#'  changepoints. Each date must be present in df$ds. If not specified,
#'  potential changepoints are selected automatically.
#' @param n.changepoints Number of potential changepoints to include. Not used
#'  if input `changepoints` is supplied. If `changepoints` is not supplied,
#'  then n.changepoints potential changepoints are selected uniformly from the
#'  first 80 percent of df$ds.
#' @param yearly.seasonality Boolean, fit yearly seasonality.
#' @param weekly.seasonality Boolean, fit weekly seasonality.
#' @param holidays data frame with columns holiday (character) and ds (date
#'  type)and optionally columns lower_window and upper_window which specify a
#'  range of days around the date to be included as holidays. lower_window=-2
#'  will include 2 days prior to the date as holidays.
#' @param seasonality.prior.scale Parameter modulating the strength of the
#'  seasonality model. Larger values allow the model to fit larger seasonal
#'  fluctuations, smaller values dampen the seasonality.
#' @param changepoint.prior.scale Parameter modulating the flexibility of the
#'  automatic changepoint selection. Large values will allow many changepoints,
#'  small values will allow few changepoints.
#' @param holidays.prior.scale Parameter modulating the strength of the holiday
#'  components model.
#' @param mcmc.samples Integer, if great than 0, will do full Bayesian
#'  inference with the specified number of MCMC samples. If 0, will do MAP
#'  estimation.
#' @param interval.width Numeric, width of the uncertainty intervals provided
#'  for the forecast. If mcmc.samples=0, this will be only the uncertainty
#'  in the trend using the MAP estimate of the extrapolated generative model.
#'  If mcmc.samples>0, this will be integrated over all model parameters,
#'  which will include uncertainty in seasonality.
#' @param uncertainty.samples Number of simulated draws used to estimate
#'  uncertainty intervals.
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
prophet <- function(df = df,
                    growth = 'linear',
                    changepoints = NULL,
                    n.changepoints = 25,
                    yearly.seasonality = TRUE,
                    weekly.seasonality = TRUE,
                    holidays = NULL,
                    seasonality.prior.scale = 10,
                    changepoint.prior.scale = 0.05,
                    holidays.prior.scale = 10,
                    mcmc.samples = 0,
                    interval.width = 0.80,
                    uncertainty.samples = 1000,
                    fit = TRUE,
                    ...
) {
  # fb-block 1

  if (!is.null(changepoints)) {
    n.changepoints <- length(changepoints)
  }

  m <- list(
    growth = growth,
    changepoints = changepoints,
    n.changepoints = n.changepoints,
    yearly.seasonality = yearly.seasonality,
    weekly.seasonality = weekly.seasonality,
    holidays = holidays,
    seasonality.prior.scale = seasonality.prior.scale,
    changepoint.prior.scale = changepoint.prior.scale,
    holidays.prior.scale = holidays.prior.scale,
    mcmc.samples = mcmc.samples,
    interval.width = interval.width,
    uncertainty.samples = uncertainty.samples,
    start = NULL,  # This and following attributes are set during fitting
    y.scale = NULL,
    t.scale = NULL,
    changepoints.t = NULL,
    stan.fit = NULL,
    params = list(),
    history = NULL
  )
  validate_inputs(m)
  class(m) <- append("prophet", class(m))
  if (fit) {
    m <- fit.prophet(m, df, ...)
  }

  # fb-block 2
  return(m)
}

#' Validates the inputs to Prophet.
#'
#' @param m Prophet object.
#'
validate_inputs <- function(m) {
  if (!(m$growth %in% c('linear', 'logistic'))) {
    stop("Parameter 'growth' should be 'linear' or 'logistic'.")
  }
  if (!is.null(m$holidays)) {
    if (!(exists('holiday', where = m$holidays))) {
      stop('Holidays dataframe must have holiday field.')
    }
    if (!(exists('ds', where = m$holidays))) {
      stop('Holidays dataframe must have ds field.')
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
      if (grepl("_delim_", h)) {
        stop('Holiday name cannot contain "_delim_"')
      }
      if (h %in% c('zeros', 'yearly', 'weekly', 'yhat', 'seasonal', 'trend')) {
        stop(paste0('Holiday name "', h, '" reserved.'))
      }
    }
  }
}

#' Load compiled Stan model
#'
#' @param model String 'linear' or 'logistic' to specify a linear or logistic
#'  trend.
#'
#' @return Stan model.
get_prophet_stan_model <- function(model) {
  fn <- paste('prophet', model, 'growth.RData', sep = '_')
  ## If the cached model doesn't work, just compile a new one.
  tryCatch({
    binary <- system.file('libs', Sys.getenv('R_ARCH'), fn,
                          package = 'prophet',
                          mustWork = TRUE)
    load(binary)
    obj.name <- paste(model, 'growth.stanm', sep = '.')
    stanm <- eval(parse(text = obj.name))

    ## Should cause an error if the model doesn't work.
    stanm@mk_cppmodule(stanm)
    stanm
  }, error = function(cond) {
    compile_stan_model(model)
  })
}

#' Compile Stan model
#'
#' @param model String 'linear' or 'logistic' to specify a linear or logistic
#'  trend.
#'
#' @return Stan model.
compile_stan_model <- function(model) {
  fn <- paste('stan/prophet', model, 'growth.stan', sep = '_')

  stan.src <- system.file(fn, package = 'prophet', mustWork = TRUE)
  stanc <- rstan::stanc(stan.src)

  model.name <- paste(model, 'growth', sep = '_')
  return(rstan::stan_model(stanc_ret = stanc, model_name = model.name))
}

#' Prepare dataframe for fitting or predicting.
#'
#' Adds a time index and scales y.
#'
#' @param m Prophet object.
#' @param df Data frame with columns ds, y, and cap if logistic growth.
#' @param initialize_scales Boolean set scaling factors in m from df.
#'
#' @return list with items 'df' and 'm'.
#'
setup_dataframe <- function(m, df, initialize_scales = FALSE) {
  if (exists('y', where=df)) {
    df$y <- as.numeric(df$y)
  }
  df$ds <- zoo::as.Date(df$ds)

  df <- df %>%
    dplyr::arrange(ds)

  if (initialize_scales) {
    m$y.scale <- max(df$y)
    m$start <- min(df$ds)
    m$t.scale <- as.numeric(max(df$ds) - m$start)
  }

  df$t <- as.numeric(df$ds - m$start) / m$t.scale
  if (exists('y', where=df)) {
    df$y_scaled <- df$y / m$y.scale
  }

  if (m$growth == 'logistic') {
    if (!(exists('cap', where=df))) {
      stop('Capacities must be supplied for logistic growth.')
    }
    df <- df %>%
      dplyr::mutate(cap_scaled = cap / m$y.scale)
  }
  return(list("m" = m, "df" = df))
}

#' Set changepoints
#'
#' Sets m$changepoints to the dates of changepoints.
#'
#' @param m Prophet object.
#'
#' @return m with changepoints set.
#'
set_changepoints <- function(m) {
  if (!is.null(m$changepoints)) {
    if (length(m$changepoints) > 0) {
      if (min(m$changepoints) < min(m$history$ds)
          || max(m$changepoints) > max(m$history$ds)) {
        stop('Changepoints must fall within training data.')
      }
    }
  } else {
    if (m$n.changepoints > 0) {
      # Place potential changepoints evenly through the first 80 pcnt of
      # the history.
      cp.indexes <- round(seq.int(1, floor(nrow(m$history) * .8),
                          length.out = (m$n.changepoints + 1))) %>%
                    utils::tail(-1)
      m$changepoints <- m$history$ds[cp.indexes]
    } else {
      m$changepoints <- c()
    }
  }
  if (length(m$changepoints) > 0) {
    m$changepoints <- zoo::as.Date(m$changepoints)
    m$changepoints.t <- sort(as.numeric(m$changepoints - m$start) / m$t.scale)
  } else {
    m$changepoints.t <- c(0)  # dummy changepoint
  }
  return(m)
}

#' Gets changepoint matrix for history dataframe.
#'
#' @param m Prophet object.
#'
#' @return array of indexes.
#'
get_changepoint_matrix <- function(m) {
  A <- matrix(0, nrow(m$history), length(m$changepoints.t))
  for (i in 1:length(m$changepoints.t)) {
    A[m$history$t >= m$changepoints.t[i], i] <- 1
  }
  return(A)
}

#' Provides fourier series components with the specified frequency.
#'
#' @param dates Vector of dates.
#' @param period Number of days of the period.
#' @param series.order Number of components.
#'
#' @return Matrix with seasonality features.
#'
fourier_series <- function(dates, period, series.order) {
  t <- dates - zoo::as.Date('1970-01-01')
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
#' @param prefix Column name prefix
#'
#' @return Dataframe with seasonality.
#'
make_seasonality_features <- function(dates, period, series.order, prefix) {
  features <- fourier_series(dates, period, series.order)
  colnames(features) <- paste(prefix, 1:ncol(features), sep = '_delim_')
  return(data.frame(features))
}

#' Construct a matrix of holiday features.
#'
#' @param m Prophet object.
#' @param dates Vector with dates used for computing seasonality.
#'
#' @return A dataframe with a column for each holiday
#'
#' @importFrom dplyr "%>%"
make_holiday_features <- function(m, dates) {
  scale.ratio <- m$holidays.prior.scale / m$seasonality.prior.scale
  wide <- m$holidays %>%
    dplyr::mutate(ds = zoo::as.Date(ds)) %>%
    dplyr::group_by(holiday, ds) %>%
    dplyr::filter(row_number() == 1) %>%
    dplyr::do({
      if (exists('lower_window', where = .) && !is.na(.$lower_window)
          && !is.na(.$upper_window)) {
        offsets <- seq(.$lower_window, .$upper_window)
      } else {
        offsets <- c(0)
      }
      names <- paste(
        .$holiday, '_delim_', ifelse(offsets < 0, '-', '+'), abs(offsets), sep = '')
      dplyr::data_frame(ds = .$ds + offsets, holiday = names)
    }) %>%
    dplyr::mutate(x = scale.ratio) %>%
    tidyr::spread(holiday, x, fill = 0)

  holiday.mat <- data.frame(ds = dates) %>%
    dplyr::left_join(wide, by = 'ds') %>%
    dplyr::select(-ds)

  holiday.mat[is.na(holiday.mat)] <- 0
  return(holiday.mat)
}

#' Data frame seasonality features.
#'
#' @param m Prophet object.
#' @param df Dataframe with dates for computing seasonality features.
#'
#' @return Dataframe with seasonality.
#'
make_all_seasonality_features <- function(m, df) {
  seasonal.features <- data.frame(zeros = rep(0, nrow(df)))
  if (m$yearly.seasonality) {
    seasonal.features <- cbind(
      seasonal.features,
      make_seasonality_features(df$ds, 365.25, 10, 'yearly'))
  }
  if (m$weekly.seasonality) {
    seasonal.features <- cbind(
      seasonal.features,
      make_seasonality_features(df$ds, 7, 3, 'weekly'))
  }
  if(!is.null(m$holidays)) {
    # A smaller prior scale will shrink holiday estimates more than seasonality
    scale.ratio <- m$holidays.prior.scale / m$seasonality.prior.scale
    seasonal.features <- cbind(
      seasonal.features,
      make_holiday_features(m, df$ds))
  }
  return(seasonal.features)
}

#' Initialize linear growth
#'
#' Provides a strong initialization for linear growth by calculating the
#' growth and offset parameters that pass the function through the first and
#' last points in the time series.
#'
#' @param df Data frame with columns ds (date), cap_scaled (scaled capacity),
#'  y_scaled (scaled time series), and t (scaled time).
#'
#' @return A vector (k, m) with the rate (k) and offset (m) of the linear
#'  growth function.
#'
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

#' Initialize logistic growth
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
logistic_growth_init <- function(df) {
  i0 <- which.min(df$ds)
  i1 <- which.max(df$ds)
  T <- df$t[i1] - df$t[i0]
  # Force valid values, in case y > cap.
  r0 <- max(1.01, df$cap_scaled[i0] / df$y_scaled[i0])
  r1 <- max(1.01, df$cap_scaled[i1] / df$y_scaled[i1])
  if (abs(r0 - r1) <= 0.01) {
    r0 <- 1.05 * r0
  }
  L0 <- log(r0 - 1)
  L1 <- log(r1 - 1)
  # Initialize the offset
  m <- L0 * T / (L0 - L1)
  # And the rate
  k <- L0 / m
  return(c(k, m))
}

#' Fit the prophet model.
#'
#' @param m Prophet object.
#' @param df Data frame.
#' @param ... Additional arguments passed to the \code{optimizing} or 
#'  \code{sampling} functions in Stan.
#'
#' @export
fit.prophet <- function(m, df, ...) {
  history <- df %>%
    dplyr::filter(!is.na(y))

  out <- setup_dataframe(m, history, initialize_scales = TRUE)
  history <- out$df
  m <- out$m
  m$history <- history
  seasonal.features <- make_all_seasonality_features(m, history)

  m <- set_changepoints(m)
  A <- get_changepoint_matrix(m)

  # Construct input to stan
  dat <- list(
    T = nrow(history),
    K = ncol(seasonal.features),
    S = length(m$changepoints.t),
    y = history$y_scaled,
    t = history$t,
    A = A,
    t_change = array(m$changepoints.t),
    X = as.matrix(seasonal.features),
    sigma = m$seasonality.prior.scale,
    tau = m$changepoint.prior.scale
  )

  # Run stan
  if (m$growth == 'linear') {
    kinit <- linear_growth_init(history)
    model <- get_prophet_stan_model('linear')
  } else {
    dat$cap <- history$cap_scaled  # Add capacities to the Stan data
    kinit <- logistic_growth_init(history)
    model <- get_prophet_stan_model('logistic')
  }

  stan_init <- function() {
    list(k = kinit[1],
         m = kinit[2],
         delta = array(rep(0, length(m$changepoints.t))),
         beta = array(rep(0, ncol(seasonal.features))),
         sigma_obs = 1
    )
  }

  if (m$mcmc.samples > 0) {
    stan.fit <- rstan::sampling(
      model,
      data = dat,
      init = stan_init,
      iter = m$mcmc.samples,
      ...
    )
    m$params <- rstan::extract(stan.fit)
    n.iteration <- length(m$params$k)
  } else {
    stan.fit <- rstan::optimizing(
      model,
      data = dat,
      init = stan_init,
      iter = 1e4,
      as_vector = FALSE,
      ...
    )
    m$params <- stan.fit$par
    n.iteration <- 1
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
#' @param df Dataframe with dates for predictions, and capacity if logistic
#'  growth. If not provided, predictions are made on the history.
#' @param ... additional arguments
#'
#' @return A data_frame with a forecast
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
  if (is.null(df)) {
    df <- object$history
  } else {
    out <- setup_dataframe(object, df)
    df <- out$df
  }

  df$trend <- predict_trend(object, df)

  df <- df %>%
    dplyr::bind_cols(predict_uncertainty(object, df)) %>%
    dplyr::bind_cols(predict_seasonal_components(object, df))
  df$yhat <- df$trend + df$seasonal
  return(df)
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
#' @param df Data frame.
#'
predict_trend <- function(model, df) {
  k <- mean(model$params$k, na.rm = TRUE)
  param.m <- mean(model$params$m, na.rm = TRUE)
  deltas <- colMeans(model$params$delta, na.rm = TRUE)

  t <- df$t
  if (model$growth == 'linear') {
    trend <- piecewise_linear(t, deltas, k, param.m, model$changepoints.t)
  } else {
    cap <- df$cap_scaled
    trend <- piecewise_logistic(
      t, cap, deltas, k, param.m, model$changepoints.t)
  }
  return(trend * model$y.scale)
}

#' Seasonality broken down into components
#'
#' @param m Prophet object.
#' @param df Data frame.
#'
predict_seasonal_components <- function(m, df) {
  seasonal.features <- make_all_seasonality_features(m, df)
  lower.p <- (1 - m$interval.width)/2
  upper.p <- (1 + m$interval.width)/2

  # Broken down into components
  components <- dplyr::data_frame(component = colnames(seasonal.features)) %>%
    dplyr::mutate(col = 1:n()) %>%
    tidyr::separate(component, c('component', 'part'), sep = "_delim_",
                    extra = "merge", fill = "right") %>%
    dplyr::filter(component != 'zeros')

  if (nrow(components) > 0) {
    component.predictions <- components %>%
      dplyr::group_by(component) %>% dplyr::do({
        comp <- (as.matrix(seasonal.features[, .$col])
                 %*% t(m$params$beta[, .$col, drop = FALSE])) * m$y.scale
        dplyr::data_frame(ix = 1:nrow(seasonal.features),
                          mean = rowMeans(comp, na.rm = TRUE),
                          lower = apply(comp, 1, stats::quantile, lower.p,
                                        na.rm = TRUE),
                          upper = apply(comp, 1, stats::quantile, upper.p,
                                        na.rm = TRUE))
      }) %>%
      tidyr::gather(stat, value, c(mean, lower, upper)) %>%
      dplyr::mutate(stat = ifelse(stat == 'mean', '', paste0('_', stat))) %>%
      tidyr::unite(component, component, stat, sep="") %>%
      tidyr::spread(component, value) %>%
      dplyr::select(-ix)

    component.predictions$seasonal <- rowSums(
      component.predictions[unique(components$component)])
  } else {
    component.predictions <- data.frame(seasonal = rep(0, nrow(df)))
  }
  return(component.predictions)
}

#' Prophet uncertainty intervals.
#'
#' @param m Prophet object.
#' @param df Data frame.
#'
predict_uncertainty <- function(m, df) {
  # Sample trend, seasonality, and yhat from the extrapolation model.
  n.iterations <- length(m$params$k)
  samp.per.iter <- max(1, ceiling(m$uncertainty.samples / n.iterations))
  nsamp <- n.iterations * samp.per.iter  # The actual number of samples

  seasonal.features <- make_all_seasonality_features(m, df)
  sim.values <- list("trend" = matrix(, nrow = nrow(df), ncol = nsamp),
                     "seasonal" = matrix(, nrow = nrow(df), ncol = nsamp),
                     "yhat" = matrix(, nrow = nrow(df), ncol = nsamp))

  for (i in 1:n.iterations) {
    # For each set of parameters from MCMC (or just 1 set for MAP),
    for (j in 1:samp.per.iter) {
      # Do a simulation with this set of parameters,
      sim <- sample_model(m, df, seasonal.features, i)
      # Store the results
      for (key in c("trend", "seasonal", "yhat")) {
        sim.values[[key]][,(i - 1) * samp.per.iter + j] <- sim[[key]]
      }
    }
  }

  # Add uncertainty estimates
  lower.p <- (1 - m$interval.width)/2
  upper.p <- (1 + m$interval.width)/2

  intervals <- cbind(
    t(apply(t(sim.values$yhat), 2, stats::quantile, c(lower.p, upper.p),
            na.rm = TRUE)),
    t(apply(t(sim.values$trend), 2, stats::quantile, c(lower.p, upper.p),
            na.rm = TRUE)),
    t(apply(t(sim.values$seasonal), 2, stats::quantile, c(lower.p, upper.p),
            na.rm = TRUE))
  ) %>% dplyr::as_data_frame()

  colnames(intervals) <- paste(rep(c('yhat', 'trend', 'seasonal'), each=2),
                               c('lower', 'upper'), sep = "_")
  return(intervals)
}

#' Simulate observations from the extrapolated generative model.
#'
#' @param m Prophet object.
#' @param df Dataframe that was fit by Prophet.
#' @param seasonal.features Data frame of seasonal features
#' @param iteration Int sampling iteration ot use parameters from.
#'
#' @return List of trend, seasonality, and yhat, each a vector like df$t.
#'
sample_model <- function(m, df, seasonal.features, iteration) {
  trend <- sample_predictive_trend(m, df, iteration)

  beta <- m$params$beta[iteration,]
  seasonal <- (as.matrix(seasonal.features) %*% beta) * m$y.scale

  sigma <- m$params$sigma_obs[iteration]
  noise <- stats::rnorm(nrow(df), mean = 0, sd = sigma) * m$y.scale

  return(list("yhat" = trend + seasonal + noise,
              "trend" = trend,
              "seasonal" = seasonal))
}

#' Simulate the trend using the extrapolated generative model.
#'
#' @param model Prophet object.
#' @param df Dataframe that was fit by Prophet.
#' @param iteration Int sampling iteration ot use parameters from.
#'
#' @return Vector of simulated trend over df$t.
#'
sample_predictive_trend <- function(model, df, iteration) {
  k <- model$params$k[iteration]
  param.m <- model$params$m[iteration]
  deltas <- model$params$delta[iteration,]

  t <- df$t
  T <- max(t)

  if (T > 1) {
    # Get the time discretization of the history
    dt <- diff(model$history$t)
    dt <- min(dt[dt > 0])
    # Number of time periods in the future
    N <- ceiling((T - 1) / dt)
    S <- length(model$changepoints.t)
    # The history had S split points, over t = [0, 1].
    # The forecast is on [1, T], and should have the same average frequency of
    # rate changes. Thus for N time periods in the future, we want an average
    # of S * (T - 1) changepoints in expectation.
    prob.change <- min(1, (S * (T - 1)) / N)
    # This calculation works for both history and df not uniformly spaced.
    n.changes <- stats::rbinom(1, N, prob.change)

    # Sample ts
    if (n.changes == 0) {
      changepoint.ts.new <- c()
    } else {
      changepoint.ts.new <- sort(stats::runif(n.changes, min = 1, max = T))
    }
  } else {
    changepoint.ts.new <- c()
    n.changes <- 0
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
  } else {
    cap <- df$cap_scaled
    trend <- piecewise_logistic(t, cap, deltas, k, param.m, changepoint.ts)
  }
  return(trend * model$y.scale)
}

#' Make dataframe with future dates for forecasting.
#'
#' @param m Prophet model object.
#' @param periods Int number of periods to forecast forward.
#' @param freq 'day', 'week', 'month', 'quarter', or 'year'.
#' @param include_history Boolean to include the historical dates in the data
#'  frame for predictions.
#'
#' @return Dataframe that extends forward from the end of m$history for the
#'  requested number of periods.
#'
#' @export
make_future_dataframe <- function(m, periods, freq = 'd',
                                  include_history = TRUE) {
  dates <- seq(max(m$history$ds), length.out = periods + 1, by = freq)
  dates <- dates[2:(periods + 1)]  # Drop the first, which is max(history$ds)
  if (include_history) {
    dates <- c(m$history$ds, dates)
  }
  return(data.frame(ds = dates))
}

#' Merge history and forecast for plotting.
#'
#' @param m Prophet object.
#' @param fcst Data frame returned by prophet predict.
#'
#' @importFrom dplyr "%>%"
df_for_plotting <- function(m, fcst) {
  # Make sure there is no y in fcst
  fcst$y <- NULL
  df <- m$history %>%
    dplyr::select(ds, y) %>%
    dplyr::full_join(fcst, by = "ds") %>%
    dplyr::arrange(ds)
  return(df)
}

#' Plot the prophet forecast.
#'
#' @param x Prophet object.
#' @param fcst Data frame returned by predict(m, df).
#' @param uncertainty Boolean indicating if the uncertainty interval for yhat
#'  should be plotted. Must be present in fcst as yhat_lower and yhat_upper.
#' @param xlabel Optional label for x-axis
#' @param ylabel Optional label for y-axis
#' @param ... additional arguments
#'
#' @return A ggplot2 plot.
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
plot.prophet <- function(x, fcst, uncertainty = TRUE, xlabel = 'ds',
                         ylabel = 'y', ...) {
  df <- df_for_plotting(x, fcst)
  gg <- ggplot2::ggplot(df, ggplot2::aes(x = ds, y = y)) +
    ggplot2::labs(x = xlabel, y = ylabel)
  if (exists('cap', where = df)) {
    gg <- gg + ggplot2::geom_line(
      ggplot2::aes(y = cap), linetype = 'dashed', na.rm = TRUE)
  }
  if (uncertainty && exists('yhat_lower', where = df)) {
    gg <- gg +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = yhat_lower, ymax = yhat_upper),
                           alpha = 0.2,
                           fill = "#0072B2",
                           na.rm = TRUE)
  }
  gg <- gg +
    ggplot2::geom_point(na.rm=TRUE) +
    ggplot2::geom_line(ggplot2::aes(y = yhat), color = "#0072B2",
                       na.rm = TRUE) +
    ggplot2::theme(aspect.ratio = 3 / 5)
  return(gg)
}

#' Plot the components of a prophet forecast.
#' Prints a ggplot2 with panels for trend, weekly and yearly seasonalities if
#' present, and holidays if present.
#'
#' @param m Prophet object.
#' @param fcst Data frame returned by predict(m, df).
#' @param uncertainty Boolean indicating if the uncertainty interval should be
#'  plotted for the trend, from fcst columns trend_lower and trend_upper.
#'
#' @return Invisibly return a list containing the plotted ggplot objects
#'
#' @export
#' @importFrom dplyr "%>%"
prophet_plot_components <- function(m, fcst, uncertainty = TRUE) {
  df <- df_for_plotting(m, fcst)
  # Plot the trend
  panels <- list(plot_trend(df, uncertainty))
  # Plot holiday components, if present.
  if (!is.null(m$holidays)) {
    panels[[length(panels) + 1]] <- plot_holidays(m, df, uncertainty)
  }
  # Plot weekly seasonality, if present
  if ("weekly" %in% colnames(df)) {
    panels[[length(panels) + 1]] <- plot_weekly(df, uncertainty)
  }
  # Plot yearly seasonality, if present
  if ("yearly" %in% colnames(df)) {
    panels[[length(panels) + 1]] <- plot_yearly(df, uncertainty)
  }
  # Make the plot.
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(layout = grid::grid.layout(length(panels),
                                                               1)))
  for (i in 1:length(panels)) {
    print(panels[[i]], vp = grid::viewport(layout.pos.row = i,
                                           layout.pos.col = 1))
  }
  return(invisible(panels))
}

#' Plot the prophet trend.
#'
#' @param df Forecast dataframe for plotting.
#' @param uncertainty Boolean to plot uncertainty intervals.
#'
#' @return A ggplot2 plot.
plot_trend <- function(df, uncertainty = TRUE) {
  gg.trend <- ggplot2::ggplot(df, ggplot2::aes(x = ds, y = trend)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE)
  if (exists('cap', where = df)) {
    gg.trend <- gg.trend + ggplot2::geom_line(ggplot2::aes(y = cap),
                                              linetype = 'dashed',
                                              na.rm = TRUE)
  }
  if (uncertainty) {
    gg.trend <- gg.trend +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = trend_lower,
                                        ymax = trend_upper),
                           alpha = 0.2,
                           fill = "#0072B2",
                           na.rm = TRUE)
  }
  return(gg.trend)
}

#' Plot the holidays component of the forecast.
#'
#' @param m Prophet model
#' @param df Forecast dataframe for plotting.
#' @param uncertainty Boolean to plot uncertainty intervals.
#'
#' @return A ggplot2 plot.
plot_holidays <- function(m, df, uncertainty = TRUE) {
  holiday.comps <- unique(m$holidays$holiday) %>% as.character()
  df.s <- data.frame(ds = df$ds,
                     holidays = rowSums(df[, holiday.comps, drop = FALSE]),
                     holidays_lower = rowSums(df[, paste0(holiday.comps,
                                                          "_lower"), drop = FALSE]),
                     holidays_upper = rowSums(df[, paste0(holiday.comps,
                                                          "_upper"), drop = FALSE]))
  # NOTE the above CI calculation is incorrect if holidays overlap in time.
  # Since it is just for the visualization we will not worry about it now.
  gg.holidays <- ggplot2::ggplot(df.s, ggplot2::aes(x = ds, y = holidays)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE)
  if (uncertainty) {
    gg.holidays <- gg.holidays +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = holidays_lower,
                                      ymax = holidays_upper),
                         alpha = 0.2,
                         fill = "#0072B2",
                         na.rm = TRUE)
  }
  return(gg.holidays)
}

#' Plot the weekly component of the forecast.
#'
#' @param df Forecast dataframe for plotting.
#' @param uncertainty Boolean to plot uncertainty intervals.
#'
#' @return A ggplot2 plot.
plot_weekly <- function(df, uncertainty = TRUE) {
  # Get weekday names in current locale
  days <- weekdays(seq.Date(as.Date('2017-01-01'), by='d', length.out=7))
  df.s <- df %>%
    dplyr::mutate(dow = factor(weekdays(ds), levels = days)) %>%
    dplyr::group_by(dow) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(dow)
  gg.weekly <- ggplot2::ggplot(df.s, ggplot2::aes(x = dow, y = weekly,
                                                  group = 1)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE) +
    ggplot2::labs(x = "Day of week")
  if (uncertainty) {
    gg.weekly <- gg.weekly +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = weekly_lower,
                                      ymax = weekly_upper),
                         alpha = 0.2,
                         fill = "#0072B2",
                         na.rm = TRUE)
  }
  return(gg.weekly)
}

#' Plot the yearly component of the forecast.
#'
#' @param df Forecast dataframe for plotting.
#' @param uncertainty Boolean to plot uncertainty intervals.
#'
#' @return A ggplot2 plot.
plot_yearly <- function(df, uncertainty = TRUE) {
  # Drop year from the dates
  df.s <- df %>%
    dplyr::mutate(doy = strftime(ds, format = "2000-%m-%d")) %>%
    dplyr::group_by(doy) %>%
    dplyr::slice(1) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(doy = zoo::as.Date(doy)) %>%
    dplyr::arrange(doy)
  gg.yearly <- ggplot2::ggplot(df.s, ggplot2::aes(x = doy, y = yearly,
                                                  group = 1)) +
    ggplot2::geom_line(color = "#0072B2", na.rm = TRUE) +
    ggplot2::scale_x_date(labels = scales::date_format('%B %d')) +
    ggplot2::labs(x = "Day of year")
  if (uncertainty) {
    gg.yearly <- gg.yearly +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = yearly_lower,
                                      ymax = yearly_upper),
                         alpha = 0.2,
                         fill = "#0072B2",
                         na.rm = TRUE)
  }
  return(gg.yearly)
}

# fb-block 3
