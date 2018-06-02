library(prophet)
context("Prophet stan model tests")

fn <- tryCatch({
  rstan::expose_stan_functions(
    rstan::stanc(file="../../inst/stan/prophet.stan")
  )
}, error = function(e) {
  rstan::expose_stan_functions(
    rstan::stanc(file=system.file("stan/prophet.stan", package="prophet"))
  )
})

DATA <- read.csv('data.csv')
N <- nrow(DATA)
train <- DATA[1:floor(N / 2), ]
future <- DATA[(ceiling(N/2) + 1):N, ]

DATA2 <- read.csv('data2.csv')

DATA$ds <- prophet:::set_date(DATA$ds)
DATA2$ds <- prophet:::set_date(DATA2$ds)

test_that("get_changepoint_matrix", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  history <- train
  m <- prophet(history, fit = FALSE)

  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  history <- out$df
  m <- out$m
  m$history <- history

  m <- prophet:::set_changepoints(m)

  cp <- m$changepoints.t

  mat <- get_changepoint_matrix(history$t, cp, nrow(history), length(cp))
  expect_equal(nrow(mat), floor(N / 2))
  expect_equal(ncol(mat), m$n.changepoints)
  # Compare to the R implementation
  A <- matrix(0, nrow(history), length(cp))
  for (i in 1:length(cp)) {
    A[history$t >= cp[i], i] <- 1
  }
  expect_true(all(A == mat))
})

test_that("get_zero_changepoints", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  history <- train
  m <- prophet(history, n.changepoints = 0, fit = FALSE)
  
  out <- prophet:::setup_dataframe(m, history, initialize_scales = TRUE)
  m <- out$m
  history <- out$df
  m$history <- history

  m <- prophet:::set_changepoints(m)
  cp <- m$changepoints.t
  
  mat <- get_changepoint_matrix(history$t, cp, nrow(history), length(cp))
  expect_equal(nrow(mat), floor(N / 2))
  expect_equal(ncol(mat), 1)
  expect_true(all(mat == 1))
})

test_that("linear_trend", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  t <- seq(0, 10)
  m <- 0
  k <- 1.0
  deltas <- c(0.5)
  changepoint.ts <- c(5)
  A <- get_changepoint_matrix(t, changepoint.ts, length(t), 1)

  y <- linear_trend(k, m, deltas, t, A, changepoint.ts)
  y.true <- c(0, 1, 2, 3, 4, 5, 6.5, 8, 9.5, 11, 12.5)
  expect_equal(y, y.true)

  t <- t[8:length(t)]
  A <- get_changepoint_matrix(t, changepoint.ts, length(t), 1)
  y.true <- y.true[8:length(y.true)]
  y <- linear_trend(k, m, deltas, t, A, changepoint.ts)
  expect_equal(y, y.true)
})

test_that("piecewise_logistic", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  t <- seq(0, 10)
  cap <- rep(10, 11)
  m <- 0
  k <- 1.0
  deltas <- c(0.5)
  changepoint.ts <- c(5)
  A <- get_changepoint_matrix(t, changepoint.ts, length(t), 1)

  y <- logistic_trend(k, m, deltas, t, cap, A, changepoint.ts, 1)
  y.true <- c(5.000000, 7.310586, 8.807971, 9.525741, 9.820138, 9.933071,
              9.984988, 9.996646, 9.999252, 9.999833, 9.999963)
  expect_equal(y, y.true, tolerance = 1e-6)
  
  t <- t[8:length(t)]
  A <- get_changepoint_matrix(t, changepoint.ts, length(t), 1)
  y.true <- y.true[8:length(y.true)]
  cap <- cap[8:length(cap)]
  y <- logistic_trend(k, m, deltas, t, cap, A, changepoint.ts, 1)
  expect_equal(y, y.true, tolerance = 1e-6)
})
