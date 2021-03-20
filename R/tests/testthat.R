# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

Sys.setenv("R_TESTS" = "")
library(testthat)
library(prophet)

# Run tests for default stan backend (currently rstan), then check if
# additional backends are available, and re-run tests using those.
Sys.setenv("R_STAN_BACKEND" = "RSTAN")
test_check("prophet")
if (tryCatch(prophet:::check_cmdstanr(), error = function(e) return(FALSE))) {
  Sys.setenv("R_STAN_BACKEND" = "CMDSTANR")
  test_check("prophet")
}
