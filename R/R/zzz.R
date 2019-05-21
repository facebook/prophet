# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

.onLoad <- function(libname, pkgname) {
  .prophet.stan.model <- get_prophet_stan_model()
  assign(
    ".prophet.stan.model",
    .prophet.stan.model,
    envir=parent.env(environment())
  )
}
