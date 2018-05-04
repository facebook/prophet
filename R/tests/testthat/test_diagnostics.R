library(prophet)
context("Prophet diagnostics tests")

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c("y", "yhat"))

DATA <- head(read.csv('data.csv'), 100)
DATA$ds <- as.Date(DATA$ds)

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
