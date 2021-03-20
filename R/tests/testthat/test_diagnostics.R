# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

library(prophet)
context("Prophet diagnostics tests")

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c("y", "yhat"))

DATA_all <- read.csv('data.csv')
DATA_all$ds <- as.Date(DATA_all$ds)
DATA <- head(DATA_all, 100)

test_that("cross_validation", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  # Calculate the number of cutoff points
  te <- max(DATA$ds)
  ts <- min(DATA$ds)
  horizon <- as.difftime(4, units = "days")
  period <- as.difftime(10, units = "days")
  initial <- as.difftime(115, units = "days")
  df.cv <- cross_validation(
    m, horizon = 4, units = "days", period = 10, initial = 115)
  expect_equal(length(unique(df.cv$cutoff)), 3)
  expect_equal(max(df.cv$ds - df.cv$cutoff), horizon)
  expect_true(as.Date(min(df.cv$cutoff)) >= ts + initial)
  dc <- diff(df.cv$cutoff)
  dc <- min(dc[dc > 0])
  expect_true(dc >= period)
  expect_true(all(df.cv$cutoff < df.cv$ds))
  # Each y in df.cv and DATA with same ds should be equal
  df.merged <- dplyr::left_join(df.cv, m$history, by="ds")
  expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
  df.cv <- cross_validation(
    m, horizon = 4, units = "days", period = 10, initial = 135)
  expect_equal(length(unique(df.cv$cutoff)), 1)
  expect_error(
    cross_validation(
      m, horizon = 10, units = "days", period = 10, initial = 140)
  )
})

test_that("cross_validation_logistic", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA
  df$cap <- 40
  m <- prophet(df, growth = 'logistic')
  df.cv <- cross_validation(
    m, horizon = 1, units = "days", period = 1, initial = 140)
  expect_equal(length(unique(df.cv$cutoff)), 2)
  expect_true(all(df.cv$cutoff < df.cv$ds))
  df.merged <- dplyr::left_join(df.cv, m$history, by="ds")
  expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
})

test_that("cross_validation_flat", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA
  m <- prophet(df, growth = 'flat')
  df.cv <- cross_validation(
    m, horizon = 1, units = "days", period = 1, initial = 140)
  expect_equal(length(unique(df.cv$cutoff)), 2)
  expect_true(all(df.cv$cutoff < df.cv$ds))
  df.merged <- dplyr::left_join(df.cv, m$history, by="ds")
  expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
})

test_that("cross_validation_extra_regressors", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA
  df$extra <- seq(0, nrow(df) - 1)
  df$is_conditional_week <- seq(0, nrow(df) - 1) %/% 7 %% 2
  m <- prophet()
  m <- add_seasonality(m, name = 'monthly', period = 30.5, fourier.order = 5)
  m <- add_seasonality(m, name = 'conditional_weekly', period = 7,
                       fourier.order = 3, prior.scale = 2.,
                       condition.name = 'is_conditional_week')
  m <- add_regressor(m, 'extra')
  m <- fit.prophet(m, df)
  df.cv <- cross_validation(
    m, horizon = 4, units = "days", period = 4, initial = 135)
  expect_equal(length(unique(df.cv$cutoff)), 2)
  period <- as.difftime(4, units = "days")
  dc <- diff(df.cv$cutoff)
  dc <- min(dc[dc > 0])
  expect_true(dc >= period)
  expect_true(all(df.cv$cutoff < df.cv$ds))
  df.merged <- dplyr::left_join(df.cv, m$history, by="ds")
  expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
})

test_that("cross_validation_default_value_check", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  df.cv1 <- cross_validation(
    m, horizon = 32, units = "days", period = 10)
  df.cv2 <- cross_validation(
    m, horizon = 32, units = 'days', period = 10, initial = 96)
  expect_equal(sum(dplyr::select(df.cv1 - df.cv2, y, yhat)), 0)
})

test_that("cross_validation_custom_cutoffs", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  # When specify a list of cutoffs the cutoff dates in df.cv1
  # are those specified
  cutoffs=c(as.Date('2012-07-31'), as.Date('2012-08-31'))
  df.cv <-cross_validation(
    m, horizon = 32, units = "days", period = 10, cutoffs=cutoffs)
  expect_equal(length(unique(df.cv$cutoff)), 2)
  # test this works ok when periods is NULL
  df.cv <-cross_validation(
    m, horizon = 32, units = "days", cutoffs=cutoffs)
  expect_equal(length(unique(df.cv$cutoff)), 2)
})

test_that("cross_validation_uncertainty_disabled", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  for (uncertainty in c(0, FALSE)) {
    m <- prophet(uncertainty.samples = uncertainty)
    m <- fit.prophet(m = m, df = DATA)
    df.cv <- cross_validation(
      m, horizon = 4, units = "days", period = 4, initial = 115)
    expected.cols <- c('y', 'ds', 'yhat', 'cutoff')
    expect_equal(expected.cols, colnames(df.cv))
    df.p <- performance_metrics(df.cv)
    expect_false('coverage' %in% colnames(df.p))
  }
})

test_that("performance_metrics", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  df_cv <- cross_validation(
    m, horizon = 4, units = "days", period = 10, initial = 90)
  # Aggregation level none
  df_none <- performance_metrics(df_cv, rolling_window = -1)
  expect_true(all(
    sort(colnames(df_none))
    == sort(c('horizon', 'mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'))
  ))
  expect_equal(nrow(df_none), 16)
  # Aggregation level 0
  df_0 <- performance_metrics(df_cv, rolling_window = 0)
  expect_equal(nrow(df_0), 4)
  expect_equal(length(unique(df_0$h)), 4)
  # Aggregation level 0.2
  df_horizon <- performance_metrics(df_cv, rolling_window = 0.2)
  expect_equal(nrow(df_horizon), 4)
  expect_equal(length(unique(df_horizon$horizon)), 4)
  # Aggregation level all
  df_all <- performance_metrics(df_cv, rolling_window = 1)
  expect_equal(nrow(df_all), 1)
  for (metric in c('mse', 'mape', 'mae', 'coverage')) {
    expect_equal(df_all[[metric]][1], mean(df_none[[metric]]))
  }
  # Custom list of metrics
  df_horizon <- performance_metrics(df_cv, metrics = c('coverage', 'mse'))
  expect_true(all(
    sort(colnames(df_horizon)) == sort(c('coverage', 'mse', 'horizon'))
  ))
  # Skip MAPE and MDAPE
  df_cv$y[1] <- 0.
  df_horizon <- performance_metrics(df_cv, metrics = c('coverage', 'mape'))
  expect_true(all(
    sort(colnames(df_horizon)) == sort(c('coverage', 'horizon'))
  ))
  df_horizon <- performance_metrics(df_cv, metrics = c('mape'))
  expect_null(df_horizon)
  # List of metrics containing non valid metrics
  expect_error(
     performance_metrics(df_cv, metrics = c('mse', 'error_metric')),
     'Valid values for metrics are: mse, rmse, mae, mape, mdape, smape, coverage'
  )
})

test_that("rolling_mean", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  x <- 0:9
  h <- 0:9
  df <- prophet:::rolling_mean_by_h(x=x, h=h, w=1, name='x')
  expect_equal(x, df$x)
  expect_equal(h, df$horizon)

  df <- prophet:::rolling_mean_by_h(x=x, h=h, w=4, name='x')
  expect_equal(x[4:10] - 1.5, df$x)
  expect_equal(3:9, df$horizon)

  h <- c(1., 2., 3., 4., 4., 4., 4., 4., 7., 7.)
  x.true <- c(1., 5., 22/3)
  h.true <- c(3., 4., 7.)
  df <- prophet:::rolling_mean_by_h(x=x, h=h, w=3, name='x')
  expect_equal(x.true, df$x)
  expect_equal(h.true, df$horizon)

  df <- prophet:::rolling_mean_by_h(x=x, h=h, w=10, name='x')
  expect_equal(c(7.), df$horizon)
  expect_equal(c(4.5), df$x)
})


test_that("rolling_median", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  x <- 0:9
  h <- 0:9
  df <- prophet:::rolling_median_by_h(x=x, h=h, w=1, name='x')
  expect_equal(x, df$x)
  expect_equal(h, df$horizon)

  df <- prophet:::rolling_median_by_h(x=x, h=h, w=4, name='x')
  x.true <- x[4:10] - 1.5
  expect_equal(x.true, df$x)
  expect_equal(3:9, df$horizon)

  h <- c(1., 2., 3., 4., 4., 4., 4., 4., 7., 7.)
  x.true <- c(1., 5., 8.)
  h.true <- c(3., 4., 7.)
  df <- prophet:::rolling_median_by_h(x=x, h=h, w=3, name='x')
  expect_equal(x.true, df$x)
  expect_equal(h.true, df$horizon)

  df <- prophet:::rolling_median_by_h(x=x, h=h, w=10, name='x')
  expect_equal(c(7.), df$horizon)
  expect_equal(c(4.5), df$x)
})


test_that("copy", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA_all
  df$cap <- 200.
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  inputs <- list(
    growth = c('linear', 'logistic', 'flat'),
    yearly.seasonality = c(TRUE, FALSE),
    weekly.seasonality = c(TRUE, FALSE),
    daily.seasonality = c(TRUE, FALSE),
    holidays = c('null', 'insert_dataframe'),
    seasonality.mode = c('additive', 'multiplicative')
  )
  products <- expand.grid(inputs)
  for (i in 1:length(products)) {
    if (products$holidays[i] == 'insert_dataframe') {
      holidays <- data.frame(ds=c('2016-12-25'), holiday=c('x'))
    } else {
      holidays <- NULL
    }
    m1 <- prophet(
      growth = as.character(products$growth[i]),
      changepoints = NULL,
      n.changepoints = 3,
      changepoint.range = 0.9,
      yearly.seasonality = products$yearly.seasonality[i],
      weekly.seasonality = products$weekly.seasonality[i],
      daily.seasonality = products$daily.seasonality[i],
      holidays = holidays,
      seasonality.prior.scale = 1.1,
      holidays.prior.scale = 1.1,
      changepoints.prior.scale = 0.1,
      mcmc.samples = 100,
      interval.width = 0.9,
      uncertainty.samples = 200,
      fit = FALSE
    )
    m1$country_holidays <- 'US'
    out <- prophet:::setup_dataframe(m1, df, initialize_scales = TRUE)
    m1 <- out$m
    m1$history <- out$df
    m1 <- prophet:::set_auto_seasonalities(m1)
    m2 <- prophet:::prophet_copy(m1)
    # Values should be copied correctly
    args <- c('growth', 'changepoints', 'n.changepoints', 'holidays', 'country_holidays',
              'seasonality.prior.scale', 'holidays.prior.scale',
              'changepoints.prior.scale', 'mcmc.samples', 'interval.width',
              'uncertainty.samples', 'seasonality.mode', 'changepoint.range')
    for (arg in args) {
      expect_equal(m1[[arg]], m2[[arg]])
    }
    expect_equal(FALSE, m2$yearly.seasonality)
    expect_equal(FALSE, m2$weekly.seasonality)
    expect_equal(FALSE, m2$daily.seasonality)
    expect_equal(m1$yearly.seasonality, 'yearly' %in% names(m2$seasonalities))
    expect_equal(m1$weekly.seasonality, 'weekly' %in% names(m2$seasonalities))
    expect_equal(m1$daily.seasonality, 'daily' %in% names(m2$seasonalities))
  }
  # Check for cutoff and custom seasonality and extra regressors
  changepoints <- seq.Date(as.Date('2012-06-15'), as.Date('2012-09-15'), by='d')
  cutoff <- as.Date('2012-07-25')
  m1 <- prophet(changepoints = changepoints)
  m1 <- add_seasonality(m1, 'custom', 10, 5)
  m1 <- add_regressor(m1, 'binary_feature')
  m1 <- fit.prophet(m1, df)
  m2 <- prophet:::prophet_copy(m1, cutoff)
  changepoints <- changepoints[changepoints < cutoff]
  expect_equal(prophet:::set_date(changepoints), m2$changepoints)
  expect_true('custom' %in% names(m2$seasonalities))
  expect_true('binary_feature' %in% names(m2$extra_regressors))
})
