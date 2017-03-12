library(prophet)
context("Prophet tests")

DATA <- read.csv('data.csv')
DATA$ds <- as.Date(DATA$ds)
N <- nrow(DATA)
train <- DATA[1:floor(N / 2), ]
future <- DATA[(ceiling(N/2) + 1):N, ]

test_that("load_models", {
  expect_error(prophet:::get_prophet_stan_model('linear'), NA)
  expect_error(prophet:::get_prophet_stan_model('logistic'), NA)
})

test_that("fit_predict", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(train)
  expect_error(predict(m, future), NA)
})

test_that("fit_predict_no_seasons", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(train, weekly.seasonality = FALSE, yearly.seasonality = FALSE)
  expect_error(predict(m, future), NA)
})

test_that("fit_predict_no_changepoints", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(train, n.changepoints = 0)
  expect_error(predict(m, future), NA)
})

test_that("fit_predict_changepoint_not_in_history", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  train_t <- dplyr::mutate(DATA, ds=zoo::as.Date(ds))
  train_t <- dplyr::filter(train_t, (ds < zoo::as.Date('2013-01-01')) | 
                                (ds > zoo::as.Date('2014-01-01')))
  future <- data.frame(ds=DATA$ds)
  m <- prophet(train_t, changepoints=c('2013-06-06'))
  expect_error(predict(m, future), NA)
})

test_that("fit_predict_duplicates", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  train2 <- train
  train2$y <- train2$y + 10
  train_t <- rbind(train, train2)
  m <- prophet(train_t)
  expect_error(predict(m, future), NA)
})

test_that("setup_dataframe", {
  history <- train
  m <- prophet(history, fit = FALSE)
  
  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  history <- out$df

  expect_true('t' %in% colnames(history))
  expect_equal(min(history$t), 0)
  expect_equal(max(history$t), 1)

  expect_true('y_scaled' %in% colnames(history))
  expect_equal(max(history$y_scaled), 1)
})

test_that("get_changepoints", {
  history <- train
  m <- prophet(history, fit = FALSE)

  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  history <- out$df
  m <- out$m
  m$history <- history

  m <- prophet:::set_changepoints(m)

  cp <- m$changepoints.t
  expect_equal(length(cp), m$n.changepoints)
  expect_true(min(cp) > 0)
  expect_true(max(cp) < N)

  mat <- prophet:::get_changepoint_matrix(m)
  expect_equal(nrow(mat), floor(N / 2))
  expect_equal(ncol(mat), m$n.changepoints)
})

test_that("get_zero_changepoints", {
  history <- train
  m <- prophet(history, n.changepoints = 0, fit = FALSE)
  
  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  m <- out$m
  history <- out$df
  m$history <- history

  m <- prophet:::set_changepoints(m)
  cp <- m$changepoints.t
  expect_equal(length(cp), 1)
  expect_equal(cp[1], 0)

  mat <- prophet:::get_changepoint_matrix(m)
  expect_equal(nrow(mat), floor(N / 2))
  expect_equal(ncol(mat), 1)
})

test_that("fourier_series_weekly", {
  mat <- prophet:::fourier_series(DATA$ds, 7, 3)
  true.values <- c(0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837,
                   -0.9009689)
  expect_equal(true.values, mat[1, ], tolerance = 1e-6)
})

test_that("fourier_series_yearly", {
  mat <- prophet:::fourier_series(DATA$ds, 365.25, 3)
  true.values <- c(0.7006152, -0.7135393, -0.9998330, 0.01827656, 0.7262249,
                   0.6874572)
  expect_equal(true.values, mat[1, ], tolerance = 1e-6)
})

test_that("growth_init", {
  history <- DATA
  history$cap <- max(history$y)
  m <- prophet(history, growth = 'logistic', fit = FALSE)

  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  m <- out$m
  history <- out$df

  params <- prophet:::linear_growth_init(history)
  expect_equal(params[1], 0.3055671, tolerance = 1e-6)
  expect_equal(params[2], 0.5307511, tolerance = 1e-6)

  params <- prophet:::logistic_growth_init(history)
  
  expect_equal(params[1], 1.507925, tolerance = 1e-6)
  expect_equal(params[2], -0.08167497, tolerance = 1e-6)
})

test_that("piecewise_linear", {
  t <- seq(0, 10)
  m <- 0
  k <- 1.0
  deltas <- c(0.5)
  changepoint.ts <- c(5)

  y <- prophet:::piecewise_linear(t, deltas, k, m, changepoint.ts)
  y.true <- c(0, 1, 2, 3, 4, 5, 6.5, 8, 9.5, 11, 12.5)
  expect_equal(y, y.true)

  t <- t[8:length(t)]
  y.true <- y.true[8:length(y.true)]
  y <- prophet:::piecewise_linear(t, deltas, k, m, changepoint.ts)
  expect_equal(y, y.true)
})

test_that("piecewise_logistic", {
  t <- seq(0, 10)
  cap <- rep(10, 11)
  m <- 0
  k <- 1.0
  deltas <- c(0.5)
  changepoint.ts <- c(5)

  y <- prophet:::piecewise_logistic(t, cap, deltas, k, m, changepoint.ts)
  y.true <- c(5.000000, 7.310586, 8.807971, 9.525741, 9.820138, 9.933071,
              9.984988, 9.996646, 9.999252, 9.999833, 9.999963)
  expect_equal(y, y.true, tolerance = 1e-6)
  
  t <- t[8:length(t)]
  y.true <- y.true[8:length(y.true)]
  cap <- cap[8:length(cap)]
  y <- prophet:::piecewise_logistic(t, cap, deltas, k, m, changepoint.ts)
  expect_equal(y, y.true, tolerance = 1e-6)
})

test_that("holidays", {
  holidays = data.frame(ds = zoo::as.Date(c('2016-12-25')),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(0))
  df <- data.frame(
    ds = seq(zoo::as.Date('2016-12-20'), zoo::as.Date('2016-12-31'), by='d'))
  m <- prophet(train, holidays = holidays, fit = FALSE)
  feats <- prophet:::make_holiday_features(m, df$ds)
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 2)
  expect_equal(sum(colSums(feats) - c(1, 1)), 0)

  holidays = data.frame(ds = zoo::as.Date(c('2016-12-25')),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(10))
  m <- prophet(train, holidays = holidays, fit = FALSE)
  feats <- prophet:::make_holiday_features(m, df$ds)
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 12)
})

test_that("fit_with_holidays", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  holidays <- data.frame(ds = zoo::as.Date(c('2012-06-06', '2013-06-06')),
                         holiday = c('seans-bday', 'seans-bday'),
                         lower_window = c(0, 0),
                         upper_window = c(1, 1))
  m <- prophet(DATA, holidays = holidays, uncertainty.samples = 0)
  expect_error(predict(m), NA)
})

test_that("make_future_dataframe", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(train)
  future <- make_future_dataframe(m, periods = 3, freq = 'd',
                                  include_history = FALSE)
  correct <- as.Date(c('2013-04-26', '2013-04-27', '2013-04-28'))
  expect_equal(future$ds, correct)

  future <- make_future_dataframe(m, periods = 3, freq = 'm',
                                  include_history = FALSE)
  correct <- as.Date(c('2013-05-25', '2013-06-25', '2013-07-25'))
  expect_equal(future$ds, correct)
})
