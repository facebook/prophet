# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

.onLoad <- function(libname, pkgname) {
  # Create environment for storing stan model
  assign("prophet_model_env", new.env(), parent.env(environment()))
  tryCatch({
    binary <- system.file(
      'libs',
      'prophet_stan_model.RData',
      package = 'prophet',
      mustWork = TRUE
    )
    load(binary)
    obj.name <- 'model.stanm'
    stanm <- eval(parse(text = obj.name))

    ## Should cause an error if the model doesn't work.
    stanm@mk_cppmodule(stanm)

    assign(
      ".prophet.stan.model",
      stanm,
      envir=prophet_model_env
    )
  },
  error = function(cond) {}
  )
}
