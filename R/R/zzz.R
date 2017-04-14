## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.

## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant
## of patent rights can be found in the PATENTS file in the same directory.

.onLoad <- function(libname, pkgname) {
  .prophet.stan.models <- list(
    "linear"=get_prophet_stan_model("linear"),
    "logistic"=get_prophet_stan_model("logistic"))
  assign(".prophet.stan.models", .prophet.stan.models,
         envir=parent.env(environment()))
}
