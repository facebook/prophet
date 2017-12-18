library(prophet)
context("Prophet metrics tests")

## Makes R CMD CHECK happy due to dplyr syntax below
globalVariables(c("y", "yhat"))

test_that("metrics", {
  # Create dummy model
  m <- prophet(readr::read_csv('data.csv', col_types=readr::cols(ds=readr::col_date(format = ""), y=readr::col_double())))
  # Create metric data
  forecast <- predict(m, NULL)
  df <- dplyr::inner_join(m$history, forecast, by="ds") %>%
    dplyr::select(y, yhat) %>%
    na.omit()
  # Check all metrics wether it is equal to its definition
  y <- df$y
  yhat <- df$yhat
  expect_equal(me(m), mean(y-yhat))
  expect_equal(mse(m), mean((y-yhat)^2))
  expect_equal(rmse(m), sqrt(mean((y-yhat)^2)))
  expect_equal(mae(m), mean(abs(y-yhat)))
  expect_equal(mpe(m), 100*mean((y-yhat)/y))
  expect_equal(mape(m), 100*mean(abs((y-yhat)/y)))
})
