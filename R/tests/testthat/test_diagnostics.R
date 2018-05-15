library(prophet)
context("Prophet diagnostics tests")

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c("y", "yhat"))

DATA_all <- read.csv('data.csv')
DATA_all$ds <- as.Date(DATA_all$ds)
DATA <- head(DATA_all, 100)

test_that("simulated_historical_forecasts", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  k <- 2
  for (p in c(1, 10)) {
    for (h in c(1, 3)) {
      df.shf <- simulated_historical_forecasts(
        m, horizon = h, units = 'days', k = k, period = p)
      # All cutoff dates should be less than ds dates
      expect_true(all(df.shf$cutoff < df.shf$ds))
      # The unique size of output cutoff should be equal to 'k'
      expect_equal(length(unique(df.shf$cutoff)), k)
      expect_equal(max(df.shf$ds - df.shf$cutoff),
                   as.difftime(h, units = 'days'))
      dc <- diff(df.shf$cutoff)
      dc <- min(dc[dc > 0])
      expect_true(dc >= as.difftime(p, units = 'days'))
      # Each y in df_shf and DATA with same ds should be equal
      df.merged <- dplyr::left_join(df.shf, m$history, by="ds")
      expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
    }
  }
})

test_that("simulated_historical_forecasts_logistic", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA
  df$cap <- 40
  m <- prophet(df, growth='logistic')
  df.shf <- simulated_historical_forecasts(
    m, horizon = 3, units = 'days', k = 2, period = 3)
  # All cutoff dates should be less than ds dates
  expect_true(all(df.shf$cutoff < df.shf$ds))
  # The unique size of output cutoff should be equal to 'k'
  expect_equal(length(unique(df.shf$cutoff)), 2)
  # Each y in df_shf and DATA with same ds should be equal
  df.merged <- dplyr::left_join(df.shf, m$history, by="ds")
  expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
})

test_that("simulated_historical_forecasts_extra_regressors", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA
  df$extra <- seq(0, nrow(df) - 1)
  m <- prophet()
  m <- add_seasonality(m, name = 'monthly', period = 30.5, fourier.order = 5)
  m <- add_regressor(m, 'extra')
  m <- fit.prophet(m, df)
  df.shf <- simulated_historical_forecasts(
    m, horizon = 3, units = 'days', k = 2, period = 3)
  # All cutoff dates should be less than ds dates
  expect_true(all(df.shf$cutoff < df.shf$ds))
  # The unique size of output cutoff should be equal to 'k'
  expect_equal(length(unique(df.shf$cutoff)), 2)
  # Each y in df_shf and DATA with same ds should be equal
  df.merged <- dplyr::left_join(df.shf, m$history, by="ds")
  expect_equal(sum((df.merged$y.x - df.merged$y.y) ** 2), 0)
})

test_that("simulated_historical_forecasts_default_value_check", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  df.shf1 <- simulated_historical_forecasts(
    m, horizon = 10, units = 'days', k = 1)
  df.shf2 <- simulated_historical_forecasts(
    m, horizon = 10, units = 'days', k = 1, period = 5)
  expect_equal(sum(dplyr::select(df.shf1 - df.shf2, y, yhat)), 0)
})

test_that("cross_validation", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  # Calculate the number of cutoff points
  te <- max(DATA$ds)
  ts <- min(DATA$ds)
  horizon <- as.difftime(4, units = "days")
  period <- as.difftime(10, units = "days")
  k <- 5
  df.cv <- cross_validation(
    m, horizon = 4, units = "days", period = 10, initial = 90)
  expect_equal(length(unique(df.cv$cutoff)), k)
  expect_equal(max(df.cv$ds - df.cv$cutoff), horizon)
  dc <- diff(df.cv$cutoff)
  dc <- min(dc[dc > 0])
  expect_true(dc >= period)
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

test_that("performance_metrics", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(DATA)
  df_cv <- cross_validation(
    m, horizon = 4, units = "days", period = 10, initial = 90)
  # Aggregation level none
  df_none <- performance_metrics(df_cv, rolling_window = 0)
  expect_true(all(
    sort(colnames(df_none))
    == sort(c('horizon', 'coverage', 'mae', 'mape', 'mse', 'rmse'))
  ))
  expect_equal(nrow(df_none), 14)
  # Aggregation level 0.2
  df_horizon <- performance_metrics(df_cv, rolling_window = 0.2)
  expect_equal(length(unique(df_horizon$horizon)), 4)
  expect_equal(nrow(df_horizon), 13)
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
})

test_that("copy", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA_all
  df$cap <- 200.
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  inputs <- list(
    growth = c('linear', 'logistic'),
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
    out <- prophet:::setup_dataframe(m1, df, initialize_scales = TRUE)
    m1 <- out$m
    m1$history <- out$df
    m1 <- prophet:::set_auto_seasonalities(m1)
    m2 <- prophet:::prophet_copy(m1)
    # Values should be copied correctly
    args <- c('growth', 'changepoints', 'n.changepoints', 'holidays',
              'seasonality.prior.scale', 'holidays.prior.scale',
              'changepoints.prior.scale', 'mcmc.samples', 'interval.width',
              'uncertainty.samples', 'seasonality.mode')
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
  changepoints <- changepoints[changepoints <= cutoff]
  expect_equal(prophet:::set_date(changepoints), m2$changepoints)
  expect_true('custom' %in% names(m2$seasonalities))
  expect_true('binary_feature' %in% names(m2$extra_regressors))
})
