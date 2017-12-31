library(prophet)
context("Prophet metrics tests")

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c("y", "yhat"))

DATA <- head(read.csv('data.csv'), 100)
DATA$ds <- as.Date(DATA$ds)

test_that("metrics_tests_using_model", {
  # Create dummy model
  m <- prophet(DATA)
  # Create metric data
  forecast <- predict(m, NULL)
  df <- na.omit(dplyr::inner_join(m$history, forecast, by="ds"))
  # Check all metrics wether it is equal to its definition
  y <- df$y
  yhat <- df$yhat
  expect_equal(me(m), mean(y-yhat))
  expect_equal(mse(m), mean((y-yhat)^2))
  expect_equal(rmse(m), sqrt(mean((y-yhat)^2)))
  expect_equal(mae(m), mean(abs(y-yhat)))
  expect_equal(mpe(m), 100*mean((y-yhat)/y))
  expect_equal(mape(m), 100*mean(abs((y-yhat)/y)))
  answer <- data.frame(
    me=me(m),
    mse=mse(m),
    rmse=rmse(m),
    mae=mae(m),
    mpe=mpe(m),
    mape=mape(m)
  )
  expect_equal(all_metrics(m), answer)
})

test_that("metrics_tests_using_simulated_historical_forecast", {
  #skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  # Create dummy model
  m <- prophet(DATA)
  # Run simulated historical forecast
  df <- simulated_historical_forecasts(m, horizon = 3, units = 'days', k = 2, period = 3)
  # Check all metrics wether it is equal to its definition
  y <- df$y
  yhat <- df$yhat
  expect_equal(me(df), mean(y-yhat))
  expect_equal(mse(df), mean((y-yhat)^2))
  expect_equal(rmse(df), sqrt(mean((y-yhat)^2)))
  expect_equal(mae(df), mean(abs(y-yhat)))
  expect_equal(mpe(df), 100*mean((y-yhat)/y))
  expect_equal(mape(df), 100*mean(abs((y-yhat)/y)))
  answer <- data.frame(
    me=me(df),
    mse=mse(df),
    rmse=rmse(df),
    mae=mae(df),
    mpe=mpe(df),
    mape=mape(df)
  )
  expect_equal(all_metrics(df), answer)
})
