library(prophet)
context("Prophet metrics tests")

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c("y", "yhat"))

test_that("metrics", {
  # Create dummy model
  m <- prophet(readr::read_csv('data.csv', col_types=readr::cols(ds=readr::col_date(format = ""), y=readr::col_double())))
  # Create metric data
  forecast <- predict(m, NULL)
  df <- na.omit(dplyr::select(forecast, y, yhat))
  # Check all metrics wether it is equal to its definition
  y <- df$y
  yhat <- df$yhat
  expect_equal(me(forecast), mean(y-yhat))
  expect_equal(mse(forecast), mean((y-yhat)^2))
  expect_equal(rmse(forecast), sqrt(mean((y-yhat)^2)))
  expect_equal(mae(forecast), mean(abs(y-yhat)))
  expect_equal(mpe(forecast), 100*mean((y-yhat)/y))
  expect_equal(mape(forecast), 100*mean(abs((y-yhat)/y)))
  answer <- data.frame(
    me=me(forecast),
    mse=mse(forecast),
    rmse=rmse(forecast),
    mae=mae(forecast),
    mpe=mpe(forecast),
    mape=mape(forecast)
  )
  expect_equal(all_metrics(forecast), answer)
})
