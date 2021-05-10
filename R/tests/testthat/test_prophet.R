# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

library(prophet)
context("Prophet tests")

DATA <- read.csv(test_path('data.csv'))
N <- nrow(DATA)
train <- DATA[1:floor(N / 2), ]
future <- DATA[(ceiling(N/2) + 1):N, ]

DATA2 <- read.csv(test_path('data2.csv'))

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
  expect_warning({
    # warning from prophet(), error from predict()
    m <- prophet(train, n.changepoints = 0)
  })
  fcst <- predict(m, future)

  expect_warning({
    m <- prophet(train, n.changepoints = 0, mcmc.samples = 100)
  })
  fcst <- predict(m, future)
})

test_that("fit_predict_changepoint_not_in_history", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  train_t <- dplyr::mutate(DATA, ds=prophet:::set_date(ds))
  train_t <- dplyr::filter(train_t,
    (ds < prophet:::set_date('2013-01-01')) |
    (ds > prophet:::set_date('2014-01-01')))
  future <- data.frame(ds=DATA$ds)
  expect_warning({
    # warning from prophet(), error from predict()
    m <- prophet(train_t, changepoints=c('2013-06-06'))
    expect_error(predict(m, future), NA)
  })
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

test_that("fit_predict_uncertainty_disabled", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  for (uncertainty in c(0, FALSE)) {
    m <- prophet(train, uncertainty.samples = uncertainty)
    fcst <- predict(m, future)
    expected.cols <- c('ds', 'trend', 'additive_terms', 'weekly', 'multiplicative_terms', 'yhat')
    expect_equal(expected.cols, colnames(fcst))
  }
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

test_that("setup_names_errors", {
  m <- prophet()
  expect_error(
    m <- add_seasonality(m, "3monthly"),
    "You have provided a name that is not syntactically valid in R, 3monthly"
  )
  expect_error(
    m <- add_regressor(m, "2monthsale"),
    "You have provided a name that is not syntactically valid in R, 2monthsale"
  )
})

test_that("logistic_floor", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  skip_on_os('mac')  # Resolves mysterious CRAN build issue
  m <- prophet(growth = 'logistic')
  history <- train
  history$floor <- 10.
  history$cap <- 80.
  future1 <- future
  future1$cap <- 80.
  future1$floor <- 10.
  m <- fit.prophet(m, history)
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
  m2 <- fit.prophet(m2, history2)
  expect_equal(m2$history$y_scaled[1], 1., tolerance = 1e-6)
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
  expect_true(max(cp) <= history$t[ceiling(0.8 * length(history$t))])
})

test_that("set_changepoint_range", {
  history <- train
  m <- prophet(history, fit = FALSE, changepoint.range = 0.4)

  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  history <- out$df
  m <- out$m
  m$history <- history

  m <- prophet:::set_changepoints(m)

  cp <- m$changepoints.t
  expect_equal(length(cp), m$n.changepoints)
  expect_true(min(cp) > 0)
  expect_true(max(cp) <= history$t[ceiling(0.4 * length(history$t))])
  expect_error(prophet(history, changepoint.range = -0.1))
  expect_error(prophet(history, changepoint.range = 2))
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

  params <- prophet:::flat_growth_init(history)
  expect_equal(params[1], 0, tolerance = 1e-6)
  expect_equal(params[2], 0.49335657, tolerance = 1e-6)

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

test_that("flat_trend", {
  t <- seq(0, 10)
  m <- 0.5
  y = prophet:::flat_trend(t, m)
  y.true <- rep(0.5, length(t))
  expect_equal(y, y.true, tolerance = 1e-6)

  t <- t[8:length(t)]
  y = prophet:::flat_trend(t, m)
  y.true <- y.true[8:length(y.true)]
  expect_equal(y, y.true, tolerance = 1e-6)
})

test_that("holidays", {
  holidays <- data.frame(ds = c('2016-12-25'),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(0))
  df <- data.frame(
    ds = seq(prophet:::set_date('2016-12-20'),
             prophet:::set_date('2016-12-31'), by='d'))
  m <- prophet(train, holidays = holidays, fit = FALSE)
  out <- prophet:::make_holiday_features(m, df$ds, m$holidays)
  feats <- out$holiday.features
  priors <- out$prior.scales
  names <- out$holiday.names
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 2)
  expect_equal(sum(colSums(feats) - c(1, 1)), 0)
  expect_true(all(priors == c(10., 10.)))
  expect_equal(names, c('xmas'))

  holidays <- data.frame(ds = c('2016-12-25'),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(10))
  m <- prophet(train, holidays = holidays, fit = FALSE)
  out <- prophet:::make_holiday_features(m, df$ds, m$holidays)
  feats <- out$holiday.features
  priors <- out$prior.scales
  names <- out$holiday.names
  expect_equal(nrow(feats), nrow(df))
  expect_equal(ncol(feats), 12)
  expect_true(all(priors == rep(10, 12)))
  expect_equal(names, c('xmas'))
  # Check prior specifications
  holidays <- data.frame(
    ds = prophet:::set_date(c('2016-12-25', '2017-12-25')),
    holiday = c('xmas', 'xmas'),
    lower_window = c(-1, -1),
    upper_window = c(0, 0),
    prior_scale = c(5., 5.)
  )
  m <- prophet(holidays = holidays, fit = FALSE)
  out <- prophet:::make_holiday_features(m, df$ds, m$holidays)
  priors <- out$prior.scales
  names <- out$holiday.names
  expect_true(all(priors == c(5., 5.)))
  expect_equal(names, c('xmas'))
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
  out <- prophet:::make_holiday_features(m, df$ds, m$holidays)
  priors <- out$prior.scales
  names <- out$holiday.names
  expect_true(all(priors == c(8, 8, 5, 5)))
  expect_true(all(sort(names) == c('seans-bday', 'xmas')))
  holidays2 <- data.frame(
    ds = prophet:::set_date(c('2012-06-06', '2013-06-06')),
    holiday = c('seans-bday', 'seans-bday'),
    lower_window = c(0, 0),
    upper_window = c(1, 1)
  )
  # manual coercions to avoid below bind_rows() warning
  holidays$holiday <- as.character(holidays$holiday)
  holidays2$holiday <- as.character(holidays2$holiday)
  holidays2 <- dplyr::bind_rows(holidays, holidays2)
  # manual factorizing to avoid above bind_rows() warning
  holidays2$holiday <- factor(holidays2$holiday)
  m <- prophet(holidays = holidays2, fit = FALSE, holidays.prior.scale = 4)
  out <- prophet:::make_holiday_features(m, df$ds, m$holidays)
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
  expect_error(prophet:::make_holiday_features(m, df$ds, m$holidays))
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

test_that("fit_with_country_holidays", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  holidays <- data.frame(ds = c('2012-06-06', '2013-06-06'),
                         holiday = c('seans-bday', 'seans-bday'),
                         lower_window = c(0, 0),
                         upper_window = c(1, 1))
  # Test with holidays and append_holidays
  m <- prophet(holidays = holidays, uncertainty.samples = 0)
  m <- add_country_holidays(m, 'US')
  m <- fit.prophet(m, DATA)
  expect_error(predict(m), NA)
  # There are training holidays missing in the test set
  train2 <- DATA %>% head(155)
  future2 <- DATA %>% tail(355)
  m <- prophet(uncertainty.samples = 0)
  m <- add_country_holidays(m, 'US')
  m <- fit.prophet(m, train2)
  expect_error(predict(m, future2), NA)
  # There are test holidays missing in the training set
  train2 <- DATA %>% tail(355)
  future2 <- DATA2
  m <- prophet(uncertainty.samples = 0)
  m <- add_country_holidays(m, 'US')
  m <- fit.prophet(m, train2)
  expect_error(predict(m, future2), NA)
  # Append_holidays with non-existing year
  max.year <- generated_holidays %>% 
    dplyr::filter(country=='US') %>%
    dplyr::select(year) %>%
    max()
  train2 <- data.frame('ds'=c(paste(max.year+1, "-01-01", sep=''),
                              paste(max.year+1, "-01-02", sep='')),
                       'y'=1)
  m <- prophet()
  m <- add_country_holidays(m, 'US')
  expect_warning(m <- fit.prophet(m, train2))
  # Append_holidays with non-existing country
  m <- prophet()
  expect_error(add_country_holidays(m, 'Utopia'))
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
  true <- list(
    period = 7, fourier.order = 3, prior.scale = 10, mode = 'additive',
    condition.name = NULL)
  for (name in names(true)) {
    expect_equal(m$seasonalities$weekly[[name]], true[[name]])
  }
  # Should be disabled due to too short history
  N.w <- 9
  train.w <- DATA[1:N.w, ]
  m <- prophet(train.w)
  expect_false('weekly' %in% names(m$seasonalities))
  # prophet warning: non-zero return code in optimizing
  m <- prophet(train.w, weekly.seasonality = TRUE)
  expect_true('weekly' %in% names(m$seasonalities))
  # Should be False due to weekly spacing
  train.w <- DATA[seq(1, nrow(DATA), 7), ]
  m <- prophet(train.w)
  expect_false('weekly' %in% names(m$seasonalities))
  m <- prophet(DATA, weekly.seasonality = 2, seasonality.prior.scale = 3)
  true <- list(
    period = 7, fourier.order = 2, prior.scale = 3, mode = 'additive',
    condition.name = NULL)
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
  true <- list(
    period = 365.25, fourier.order = 10, prior.scale = 10, mode = 'additive',
    condition.name = NULL)
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
  true <- list(
    period = 365.25, fourier.order = 7, prior.scale = 3, mode = 'additive',
    condition.name = NULL)
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
  true <- list(
    period = 1, fourier.order = 4, prior.scale = 10, mode = 'additive',
    condition.name = NULL)
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
  true <- list(
    period = 1, fourier.order = 7, prior.scale = 3, mode = 'additive',
    condition.name = NULL)
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
  expect_error(
    add_seasonality(m, name="incorrect.fourier.order", period=30, fourier.order=-10),
    "Fourier order must be > 0."
  )
  m <- add_seasonality(m, name='monthly', period=30, fourier.order=5)
  true <- list(
    period = 30, fourier.order = 5, prior.scale = 10, mode = 'additive',
    condition.name = NULL)
  for (name in names(true)) {
    expect_equal(m$seasonalities$monthly[[name]], true[[name]])
  }
  expect_error(
    add_seasonality(m, name='special_day', period=30, fourier.order=5),
    "already used for a holiday."
  )
  expect_error(
    add_seasonality(m, name='trend', period=30, fourier.order=5),
    "is reserved."
  )
  m <- add_seasonality(m, name='weekly', period=30, fourier.order=5)
  # Test priors
  m <- prophet(
    holidays = holidays, yearly.seasonality = FALSE,
    seasonality.mode = 'multiplicative')
  m <- add_seasonality(
    m, name='monthly', period=30, fourier.order=5, prior.scale = 2,
    mode = 'additive')
  m <- fit.prophet(m, DATA)
  expect_equal(m$seasonalities$monthly$mode, 'additive')
  expect_equal(m$seasonalities$weekly$mode, 'multiplicative')
  out <- prophet:::make_all_seasonality_features(m, m$history)
  prior.scales <- out$prior.scales
  component.cols <- out$component.cols
  expect_equal(sum(component.cols$monthly), 10)
  expect_equal(sum(component.cols$special_day), 1)
  expect_equal(sum(component.cols$weekly), 6)
  expect_equal(sum(component.cols$additive_terms), 10)
  expect_equal(sum(component.cols$multiplicative_terms), 7)
  expect_equal(sum(component.cols$monthly[1:11]), 10)
  expect_equal(sum(component.cols$weekly[11:17]), 6)
  expect_true(all(prior.scales == c(rep(2, 10), rep(10, 6), 4)))
})

test_that("conditional_custom_seasonality", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet(weekly_seasonality=FALSE, yearly_seasonality=FALSE)
  m <- add_seasonality(m, name='conditional_weekly', period=7, fourier.order=3,
                       prior.scale=2., condition.name='is_conditional_week')
  m <- add_seasonality(m, name='normal_monthly', period=30.5, fourier.order=5,
                       prior.scale=2.)
  df <- DATA
  # Require all conditions names in df
  expect_error(
    fit.prophet(m, df)
  )
  df$is_conditional_week <- c(rep(0, 255), rep(2, 255))
  # Require boolean compatible values
  expect_error(
    fit.prophet(m, df)
  )
  df$is_conditional_week <- c(rep(0, 255), rep(1, 255))
  fit.prophet(m, df)
  true <- list(
    period = 7, fourier.order = 3, prior.scale = 2., mode = 'additive', 
    condition.name = 'is_conditional_week')
  for (name in names(true)) {
    expect_equal(m$seasonalities[['conditional_weekly']][[name]], true[[name]])
  }
  expect_equal(
    m$seasonalities[['normal_monthly']]$condition.name, NULL
  )
  out <- prophet:::make_all_seasonality_features(m, df)
  #Confirm that only values without is_conditional_week has non zero entries
  nonzero.weekly = out$seasonal.features %>%
    dplyr::as_tibble() %>%
    dplyr::select(dplyr::starts_with('conditional_weekly')) %>%
    dplyr::mutate_all(~ . != 0) %>%
    dplyr::mutate(nonzero = rowSums(. != 0) > 0) %>% 
    dplyr::pull(nonzero)
  expect_equal(
    nonzero.weekly, as.logical(df$is_conditional_week)
  )
})

test_that("added_regressors", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet()
  m <- add_regressor(m, 'binary_feature', prior.scale=0.2)
  m <- add_regressor(m, 'numeric_feature', prior.scale=0.5)
  m <- add_regressor(
    m, 'numeric_feature2', prior.scale=0.5, mode = 'multiplicative')
  m <- add_regressor(m, 'binary_feature2', standardize=TRUE)
  df <- DATA
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  df$numeric_feature <- 0:509
  df$numeric_feature2 <- 0:509
  # Require all regressors in df
  expect_error(
    fit.prophet(m, df)
  )
  df$binary_feature2 <- c(rep(1, 100), rep(0, 410))
  m <- fit.prophet(m, df)
  # Check that standardizations are correctly set
  true <- list(
    prior.scale = 0.2, mu = 0, std = 1, standardize = 'auto', mode = 'additive'
  )
  for (name in names(true)) {
    expect_equal(true[[name]], m$extra_regressors$binary_feature[[name]])
  }
  true <- list(prior.scale = 0.5, mu = 254.5, std = 147.368585)
  for (name in names(true)) {
    expect_equal(true[[name]], m$extra_regressors$numeric_feature[[name]],
                 tolerance = 1e-5)
  }
  expect_equal(m$extra_regressors$numeric_feature2$mode, 'multiplicative')
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
  component.cols <- out$component.cols
  modes <- out$modes
  expect_equal(ncol(seasonal.features), 30)
  r_names <- c('binary_feature', 'numeric_feature', 'binary_feature2')
  true.priors <- c(0.2, 0.5, 10.)
  for (i in seq_along(r_names)) {
    name <- r_names[i]
    expect_true(name %in% colnames(seasonal.features))
    expect_equal(sum(component.cols[[name]]), 1)
    expect_equal(sum(prior.scales * component.cols[[name]]), true.priors[i])
  }
  # Check that forecast components are reasonable
  future <- data.frame(
    ds = c('2014-06-01'),
    binary_feature = c(0),
    numeric_feature = c(10),
    numeric_feature2 = c(10)
  )
  expect_error(predict(m, future))
  future$binary_feature2 <- 0.
  fcst <- predict(m, future)
  expect_equal(ncol(fcst), 37)
  expect_equal(fcst$binary_feature[1], 0)
  expect_equal(fcst$extra_regressors_additive[1],
               fcst$numeric_feature[1] + fcst$binary_feature2[1])
  expect_equal(fcst$extra_regressors_multiplicative[1],
               fcst$numeric_feature2[1])
  expect_equal(fcst$additive_terms[1],
               fcst$yearly[1] + fcst$weekly[1]
               + fcst$extra_regressors_additive[1])
  expect_equal(fcst$multiplicative_terms[1],
               fcst$extra_regressors_multiplicative[1])
  expect_equal(
    fcst$yhat[1],
    fcst$trend[1] * (1 + fcst$multiplicative_terms[1]) + fcst$additive_terms[1]
  )
  # Check works with constant extra regressor of 0
  df$constant_feature <- 0
  m <- prophet()
  m <- add_regressor(m, 'constant_feature', standardize = TRUE)
  m <- fit.prophet(m, df)
  expect_equal(m$extra_regressors$constant_feature$std, 1)
})

test_that("set_seasonality_mode", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- prophet()
  expect_equal(m$seasonality.mode, 'additive')
  m <- prophet(seasonality.mode = 'multiplicative')
  expect_equal(m$seasonality.mode, 'multiplicative')
  expect_error(prophet(seasonality.mode = 'batman'))
})

test_that("seasonality_modes", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  holidays <- data.frame(ds = c('2016-12-25'),
                        holiday = c('xmas'),
                        lower_window = c(-1),
                        upper_window = c(0))
  m <- prophet(seasonality.mode = 'multiplicative', holidays = holidays)
  m <- add_seasonality(
    m, name = 'monthly', period = 30, fourier.order = 3, mode = 'additive')
  m <- add_regressor(m, name = 'binary_feature', mode = 'additive')
  m <- add_regressor(m, name = 'numeric_feature')
  # Construct seasonal features
  df <- DATA
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  df$numeric_feature <- 0:509
  out <- prophet:::setup_dataframe(m, df, initialize_scales = TRUE)
  df <- out$df
  m <- out$m
  m$history <- df
  m <- prophet:::set_auto_seasonalities(m)
  out <- prophet:::make_all_seasonality_features(m, df)
  component.cols <- out$component.cols
  modes <- out$modes
  expect_equal(sum(component.cols$additive_terms), 7)
  expect_equal(sum(component.cols$multiplicative_terms), 29)
  expect_equal(
    sort(modes$additive),
    c('additive_terms', 'binary_feature', 'extra_regressors_additive',
      'monthly')
  )
  expect_equal(
    sort(modes$multiplicative),
    c('extra_regressors_multiplicative', 'holidays', 'multiplicative_terms',
      'numeric_feature', 'weekly', 'xmas', 'yearly')
  )
})
