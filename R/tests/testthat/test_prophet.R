library(prophet)
context("Prophet tests")

DATA <- read.csv('data.csv')
N <- nrow(DATA)
train <- DATA[1:floor(N / 2), ]
future <- DATA[(ceiling(N/2) + 1):N, ]

DATA2 <- read.csv('data2.csv')

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
  train_t <- dplyr::mutate(DATA, ds=prophet:::set_date(ds))
  train_t <- dplyr::filter(train_t,
    (ds < prophet:::set_date('2013-01-01')) |
    (ds > prophet:::set_date('2014-01-01')))
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
  true.values <- c(0.9165623, 0.3998920, 0.7330519, -0.6801727, -0.3302791,
                   -0.9438833)
  expect_equal(true.values, mat[1, ], tolerance = 1e-6)
})

test_that("fourier_series_yearly", {
  mat <- prophet:::fourier_series(DATA$ds, 365.25, 3)
  true.values <- c(0.69702635, -0.71704551, -0.99959923, 0.02830854,
                   0.73648994, 0.67644849)
  expect_equal(true.values, mat[1, ], tolerance = 1e-6)
})

test_that("growth_init", {
  history <- DATA[1:468, ]
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
  holidays = data.frame(ds = c('2016-12-25'),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(0))
  df <- data.frame(
    ds = seq(prophet:::set_date('2016-12-20'),
             prophet:::set_date('2016-12-31'), by='d'))
  m <- prophet(train, holidays = holidays, fit = FALSE)
  feats <- prophet:::make_holiday_features(m, df$ds)
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 2)
  expect_equal(sum(colSums(feats) - c(1, 1)), 0)

  holidays = data.frame(ds = c('2016-12-25'),
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
  holidays <- data.frame(ds = c('2012-06-06', '2013-06-06'),
                         holiday = c('seans-bday', 'seans-bday'),
                         lower_window = c(0, 0),
                         upper_window = c(1, 1))
  m <- prophet(DATA, holidays = holidays, uncertainty.samples = 0)
  expect_error(predict(m), NA)
})

test_that("make_future_dataframe", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  train.t <- DATA[1:234, ]
  m <- prophet(train.t)
  future <- make_future_dataframe(m, periods = 3, freq = 'day',
                                  include_history = FALSE)
  correct <- prophet:::set_date(c('2013-04-26', '2013-04-27', '2013-04-28'))
  expect_equal(future$ds, correct)

  future <- make_future_dataframe(m, periods = 3, freq = 'month',
                                  include_history = FALSE)
  correct <- prophet:::set_date(c('2013-05-25', '2013-06-25', '2013-07-25'))
  expect_equal(future$ds, correct)
})

test_that("auto_weekly_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  # Should be enabled
  N.w <- 15
  train.w <- DATA[1:N.w, ]
  m <- prophet(train.w, fit = FALSE)
  expect_equal(m$weekly.seasonality, 'auto')
  m <- prophet:::fit.prophet(m, train.w)
  expect_true('weekly' %in% names(m$seasonalities))
  expect_equal(m$seasonalities[['weekly']], c(7, 3))
  # Should be disabled due to too short history
  N.w <- 9
  train.w <- DATA[1:N.w, ]
  m <- prophet(train.w)
  expect_false('weekly' %in% names(m$seasonalities))
  m <- prophet(train.w, weekly.seasonality = TRUE)
  expect_true('weekly' %in% names(m$seasonalities))
  # Should be False due to weekly spacing
  train.w <- DATA[seq(1, nrow(DATA), 7), ]
  m <- prophet(train.w)
  expect_false('weekly' %in% names(m$seasonalities))
  m <- prophet(DATA, weekly.seasonality=2)
  expect_equal(m$seasonalities[['weekly']], c(7, 2))
})

test_that("auto_yearly_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  # Should be enabled
  m <- prophet(DATA, fit = FALSE)
  expect_equal(m$yearly.seasonality, 'auto')
  m <- prophet:::fit.prophet(m, DATA)
  expect_true('yearly' %in% names(m$seasonalities))
  expect_equal(m$seasonalities[['yearly']], c(365.25, 10))
  # Should be disabled due to too short history
  N.w <- 240
  train.y <- DATA[1:N.w, ]
  m <- prophet(train.y)
  expect_false('yearly' %in% names(m$seasonalities))
  m <- prophet(train.y, yearly.seasonality = TRUE)
  expect_true('yearly' %in% names(m$seasonalities))
  m <- prophet(DATA, yearly.seasonality=7)
  expect_equal(m$seasonalities[['yearly']], c(365.25, 7))
})

test_that("auto_daily_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  # Should be enabled
  m <- prophet(DATA2, fit = FALSE)
  expect_equal(m$daily.seasonality, 'auto')
  m <- prophet:::fit.prophet(m, DATA2)
  expect_true('daily' %in% names(m$seasonalities))
  expect_equal(m$seasonalities[['daily']], c(1, 4))
  # Should be disabled due to too short history
  N.d <- 430
  train.y <- DATA2[1:N.d, ]
  m <- prophet(train.y)
  expect_false('daily' %in% names(m$seasonalities))
  m <- prophet(train.y, daily.seasonality = TRUE)
  expect_true('daily' %in% names(m$seasonalities))
  m <- prophet(DATA2, daily.seasonality=7)
  expect_equal(m$seasonalities[['daily']], c(1, 7))
  m <- prophet(DATA)
  expect_false('daily' %in% names(m$seasonalities))
})

test_that("test_subdaily_holidays", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  holidays <- data.frame(ds = c('2017-01-02'),
                         holiday = c('special_day'))
  m <- prophet(DATA2, holidays=holidays)
  fcst <- predict(m)
  expect_equal(sum(fcst$special_day == 0), 575)
})

test_that("custom_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  holidays <- data.frame(ds = c('2017-01-02'),
                         holiday = c('special_day'))
  m <- prophet(holidays=holidays)
  m <- add_seasonality(m, name='monthly', period=30, fourier.order=5)
  expect_equal(m$seasonalities[['monthly']], c(30, 5))
})
