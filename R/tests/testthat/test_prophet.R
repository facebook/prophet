library(prophet)
context("Prophet tests")

DATA <- read.csv('data.csv')
N <- nrow(DATA)
train <- DATA[1:floor(N / 2), ]
future <- DATA[(ceiling(N/2) + 1):N, ]

DATA2 <- read.csv('data2.csv')

DATA$ds <- prophet:::set_date(DATA$ds)
DATA2$ds <- prophet:::set_date(DATA2$ds)

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

test_that("fit_predict_constant_history", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  train2 <- train
  train2$y <- 20
  m <- prophet(train2)
  fcst <- predict(m, future)
  expect_equal(tail(fcst$yhat, 1), 20)
  train2$y <- 0
  m <- prophet(train2)
  fcst <- predict(m, future)
  expect_equal(tail(fcst$yhat, 1), 0)
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

test_that("logistic_floor", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(growth = 'logistic')
  history <- train
  history$floor <- 10.
  history$cap <- 80.
  future1 <- future
  future1$cap <- 80.
  future1$floor <- 10.
  m <- fit.prophet(m, history, algorithm = 'Newton')
  expect_true(m$logistic.floor)
  expect_true('floor' %in% colnames(m$history))
  expect_equal(m$history$y_scaled[1], 1., tolerance = 1e-6)
  fcst1 <- predict(m, future1)

  m2 <- prophet(growth = 'logistic')
  history2 <- history
  history2$y <- history2$y + 10.
  history2$floor <- history2$floor + 10.
  history2$cap <- history2$cap + 10.
  future1$cap <- future1$cap + 10.
  future1$floor <- future1$floor + 10.
  m2 <- fit.prophet(m2, history2, algorithm = 'Newton')
  expect_equal(m2$history$y_scaled[1], 1., tolerance = 1e-6)
  fcst2 <- predict(m, future1)
  fcst2$yhat <- fcst2$yhat - 10.
  # Check for approximate shift invariance
  expect_true(all(abs(fcst1$yhat - fcst2$yhat) < 1))
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

test_that("override_n_changepoints", {
  history <- train[1:20,]
  m <- prophet(history, fit = FALSE)

  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  m <- out$m
  history <- out$df
  m$history <- history

  m <- prophet:::set_changepoints(m)
  expect_equal(m$n.changepoints, 15)
  cp <- m$changepoints.t
  expect_equal(length(cp), 15)
})

test_that("fourier_series_weekly", {
  true.values <- c(0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837,
                   -0.9009689)
  mat <- prophet:::fourier_series(DATA$ds, 7, 3)
    expect_equal(true.values, mat[1, ], tolerance = 1e-6)
})

test_that("fourier_series_yearly", {
  true.values <- c(0.7006152, -0.7135393, -0.9998330, 0.01827656, 0.7262249,
                   0.6874572)
  mat <- prophet:::fourier_series(DATA$ds, 365.25, 3)
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
  out <- prophet:::make_holiday_features(m, df$ds)
  feats <- out$holiday.features
  priors <- out$prior.scales
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 2)
  expect_equal(sum(colSums(feats) - c(1, 1)), 0)
  expect_true(all(priors == c(10., 10.)))

  holidays = data.frame(ds = c('2016-12-25'),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(10))
  m <- prophet(train, holidays = holidays, fit = FALSE)
  out <- prophet:::make_holiday_features(m, df$ds)
  feats <- out$holiday.features
  priors <- out$prior.scales
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 12)
  expect_true(all(priors == rep(10, 12)))
  # Check prior specifications
  holidays <- data.frame(
    ds = prophet:::set_date(c('2016-12-25', '2017-12-25')),
    holiday = c('xmas', 'xmas'),
    lower_window = c(-1, -1),
    upper_window = c(0, 0),
    prior_scale = c(5., 5.)
  )
  m <- prophet(holidays = holidays, fit = FALSE)
  out <- prophet:::make_holiday_features(m, df$ds)
  priors <- out$prior.scales
  expect_true(all(priors == c(5., 5.)))
  # 2 different priors
  holidays2 <- data.frame(
    ds = prophet:::set_date(c('2012-06-06', '2013-06-06')),
    holiday = c('seans-bday', 'seans-bday'),
    lower_window = c(0, 0),
    upper_window = c(1, 1),
    prior_scale = c(8, 8)
  )
  holidays2 <- rbind(holidays, holidays2)
  m <- prophet(holidays = holidays2, fit = FALSE)
  out <- prophet:::make_holiday_features(m, df$ds)
  priors <- out$prior.scales
  expect_true(all(priors == c(8, 8, 5, 5)))
  holidays2 <- data.frame(
    ds = prophet:::set_date(c('2012-06-06', '2013-06-06')),
    holiday = c('seans-bday', 'seans-bday'),
    lower_window = c(0, 0),
    upper_window = c(1, 1)
  )
  holidays2 <- dplyr::bind_rows(holidays, holidays2)
  m <- prophet(holidays = holidays2, fit = FALSE, holidays.prior.scale = 4)
  out <- prophet:::make_holiday_features(m, df$ds)
  priors <- out$prior.scales
  expect_true(all(priors == c(4, 4, 5, 5)))
  # Check incompatible priors
  holidays <- data.frame(
    ds = prophet:::set_date(c('2016-12-25', '2016-12-27')),
    holiday = c('xmasish', 'xmasish'),
    lower_window = c(-1, -1),
    upper_window = c(0, 0),
    prior_scale = c(5., 6.)
  )
  m <- prophet(holidays = holidays, fit = FALSE)
  expect_error(prophet:::make_holiday_features(m, df$ds))
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
  m <- fit.prophet(m, train.w)
  expect_true('weekly' %in% names(m$seasonalities))
  true <- list(period = 7, fourier.order = 3, prior.scale = 10)
  for (name in names(true)) {
    expect_equal(m$seasonalities$weekly[[name]], true[[name]])
  }
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
  m <- prophet(DATA, weekly.seasonality = 2, seasonality.prior.scale = 3)
  true <- list(period = 7, fourier.order = 2, prior.scale = 3)
  for (name in names(true)) {
    expect_equal(m$seasonalities$weekly[[name]], true[[name]])
  }
})

test_that("auto_yearly_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  # Should be enabled
  m <- prophet(DATA, fit = FALSE)
  expect_equal(m$yearly.seasonality, 'auto')
  m <- fit.prophet(m, DATA)
  expect_true('yearly' %in% names(m$seasonalities))
  true <- list(period = 365.25, fourier.order = 10, prior.scale = 10)
  for (name in names(true)) {
    expect_equal(m$seasonalities$yearly[[name]], true[[name]])
  }
  # Should be disabled due to too short history
  N.w <- 240
  train.y <- DATA[1:N.w, ]
  m <- prophet(train.y)
  expect_false('yearly' %in% names(m$seasonalities))
  m <- prophet(train.y, yearly.seasonality = TRUE)
  expect_true('yearly' %in% names(m$seasonalities))
  m <- prophet(DATA, yearly.seasonality = 7, seasonality.prior.scale = 3)
  true <- list(period = 365.25, fourier.order = 7, prior.scale = 3)
  for (name in names(true)) {
    expect_equal(m$seasonalities$yearly[[name]], true[[name]])
  }
})

test_that("auto_daily_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  # Should be enabled
  m <- prophet(DATA2, fit = FALSE)
  expect_equal(m$daily.seasonality, 'auto')
  m <- fit.prophet(m, DATA2)
  expect_true('daily' %in% names(m$seasonalities))
  true <- list(period = 1, fourier.order = 4, prior.scale = 10)
  for (name in names(true)) {
    expect_equal(m$seasonalities$daily[[name]], true[[name]])
  }
  # Should be disabled due to too short history
  N.d <- 430
  train.y <- DATA2[1:N.d, ]
  m <- prophet(train.y)
  expect_false('daily' %in% names(m$seasonalities))
  m <- prophet(train.y, daily.seasonality = TRUE)
  expect_true('daily' %in% names(m$seasonalities))
  m <- prophet(DATA2, daily.seasonality = 7, seasonality.prior.scale = 3)
  true <- list(period = 1, fourier.order = 7, prior.scale = 3)
  for (name in names(true)) {
    expect_equal(m$seasonalities$daily[[name]], true[[name]])
  }
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
                         holiday = c('special_day'),
                         prior_scale = c(4))
  m <- prophet(holidays=holidays)
  m <- add_seasonality(m, name='monthly', period=30, fourier.order=5)
  true <- list(period = 30, fourier.order = 5, prior.scale = 10)
  for (name in names(true)) {
    expect_equal(m$seasonalities$monthly[[name]], true[[name]])
  }
  expect_error(
    add_seasonality(m, name='special_day', period=30, fourier_order=5)
  )
  expect_error(
    add_seasonality(m, name='trend', period=30, fourier_order=5)
  )
  m <- add_seasonality(m, name='weekly', period=30, fourier.order=5)
  # Test priors
  m <- prophet(holidays = holidays, yearly.seasonality = FALSE)
  m <- add_seasonality(
    m, name='monthly', period=30, fourier.order=5, prior.scale = 2)
  m <- fit.prophet(m, DATA)
  prior.scales <- prophet:::make_all_seasonality_features(
    m, m$history)$prior.scales
  expect_true(all(prior.scales == c(rep(2, 10), rep(10, 6), 4)))
})

test_that("added_regressors", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet()
  m <- add_regressor(m, 'binary_feature', prior.scale=0.2)
  m <- add_regressor(m, 'numeric_feature', prior.scale=0.5)
  m <- add_regressor(m, 'binary_feature2', standardize=TRUE)
  df <- DATA
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  df$numeric_feature <- 0:509
  # Require all regressors in df
  expect_error(
    fit.prophet(m, df)
  )
  df$binary_feature2 <- c(rep(1, 100), rep(0, 410))
  m <- fit.prophet(m, df)
  # Check that standardizations are correctly set
  true <- list(prior.scale = 0.2, mu = 0, std = 1, standardize = 'auto')
  for (name in names(true)) {
    expect_equal(true[[name]], m$extra_regressors$binary_feature[[name]])
  }
  true <- list(prior.scale = 0.5, mu = 254.5, std = 147.368585)
  for (name in names(true)) {
    expect_equal(true[[name]], m$extra_regressors$numeric_feature[[name]],
                 tolerance = 1e-5)
  }
  true <- list(prior.scale = 10., mu = 0.1960784, std = 0.3974183)
  for (name in names(true)) {
    expect_equal(true[[name]], m$extra_regressors$binary_feature2[[name]],
                 tolerance = 1e-5)
  }
  # Check that standardization is done correctly
  df2 <- prophet:::setup_dataframe(m, df)$df
  expect_equal(df2$binary_feature[1], 0)
  expect_equal(df2$numeric_feature[1], -1.726962, tolerance = 1e-4)
  expect_equal(df2$binary_feature2[1], 2.022859, tolerance = 1e-4)
  # Check that feature matrix and prior scales are correctly constructed
  out <- prophet:::make_all_seasonality_features(m, df2)
  seasonal.features <- out$seasonal.features
  prior.scales <- out$prior.scales
  expect_true('binary_feature' %in% colnames(seasonal.features))
  expect_true('numeric_feature' %in% colnames(seasonal.features))
  expect_true('binary_feature2' %in% colnames(seasonal.features))
  expect_equal(ncol(seasonal.features), 29)
  expect_true(all(sort(prior.scales[27:29]) == c(0.2, 0.5, 10.)))
  # Check that forecast components are reasonable
  future <- data.frame(
    ds = c('2014-06-01'), binary_feature = c(0), numeric_feature = c(10))
  expect_error(predict(m, future))
  future$binary_feature2 <- 0.
  fcst <- predict(m, future)
  expect_equal(ncol(fcst), 31)
  expect_equal(fcst$binary_feature[1], 0)
  expect_equal(fcst$extra_regressors[1],
               fcst$numeric_feature[1] + fcst$binary_feature2[1])
  expect_equal(fcst$seasonalities[1], fcst$yearly[1] + fcst$weekly[1])
  expect_equal(fcst$seasonal[1],
               fcst$seasonalities[1] + fcst$extra_regressors[1])
  expect_equal(fcst$yhat[1], fcst$trend[1] + fcst$seasonal[1])
  # Check fails if constant extra regressor
  df$constant_feature <- 5
  m <- prophet()
  m <- add_regressor(m, 'constant_feature')
  expect_error(fit.prophet(m, df))
})

test_that("copy", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  df <- DATA
  df$cap <- 200.
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  inputs <- list(
    growth = c('linear', 'logistic'),
    yearly.seasonality = c(TRUE, FALSE),
    weekly.seasonality = c(TRUE, FALSE),
    daily.seasonality = c(TRUE, FALSE),
    holidays = c('null', 'insert_dataframe')
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
              'uncertainty.samples')
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
