library(prophet)
context("Prophet stan model tests")

rstan::expose_stan_functions(rstan::stanc(file="../..//inst/stan/prophet_logistic_growth.stan"))

DATA <- read.csv('data.csv')
N <- nrow(DATA)
train <- DATA[1:floor(N / 2), ]
future <- DATA[(ceiling(N/2) + 1):N, ]

DATA2 <- read.csv('data2.csv')

DATA$ds <- prophet:::set_date(DATA$ds)
DATA2$ds <- prophet:::set_date(DATA2$ds)

test_that("get_changepoint_matrix", {
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
