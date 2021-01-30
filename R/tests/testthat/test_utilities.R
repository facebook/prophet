# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

library(prophet)
context("Prophet utilities tests")

DATA <- read.csv('data.csv')
DATA$ds <- as.Date(DATA$ds)

build_model_with_regressors <- function(data, mcmc.samples = 0) {
  m <- prophet(mcmc.samples = mcmc.samples)
  m <- add_regressor(m, 'binary_feature', prior.scale=0.2)
  m <- add_regressor(m, 'numeric_feature', prior.scale=0.5)
  m <- add_regressor(
    m, 'numeric_feature2', prior.scale=0.5, mode = 'multiplicative')
  m <- add_regressor(m, 'binary_feature2', standardize=TRUE)

  df <- data
  df$binary_feature <- c(rep(0, 255), rep(1, 255))
  df$numeric_feature <- 0:509
  df$numeric_feature2 <- 0:509
  df$binary_feature2 <- c(rep(1, 100), rep(0, 410))
  m <- fit.prophet(m, df)

  return(m)
}

test_that("regressor_coefficients_no_uncertainty", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  m <- build_model_with_regressors(DATA, mcmc.samples = 0)
  coefs <- regressor_coefficients(m)

  expect_equal(dim(coefs), c(4, 6))
  expect_equal(coefs[, "coef_lower"], coefs[, "coef"])
  expect_equal(coefs[, "coef_upper"], coefs[, "coef"])
})

test_that("regressor_coefficients_with_uncertainty", {
  skip_if_not(Sys.getenv('R_ARCH') != '/i386')
  suppressWarnings(m <- build_model_with_regressors(DATA, mcmc.samples = 100))
  coefs <- regressor_coefficients(m)

  expect_equal(dim(coefs), c(4, 6))
  expect_true(all(coefs[, "coef_lower"] < coefs[, "coef"]))
  expect_true(all(coefs[, "coef_upper"] > coefs[, "coef"]))
})
