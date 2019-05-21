# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

packageStartupMessage('Compiling model (this will take a minute...)')

dest <- file.path(R_PACKAGE_DIR, paste0('libs', R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)

packageStartupMessage(paste('Writing model to:', dest))
packageStartupMessage(paste('Compiling using binary:', R.home('bin')))

model.src <- file.path(R_PACKAGE_SOURCE, 'inst', 'stan', 'prophet.stan')
model.binary <- file.path(dest, 'prophet_stan_model.RData')

# See: https://github.com/r-lib/pkgbuild/issues/54#issuecomment-448702834
# TODO: move stan compilation into Makevars
suppressMessages({
  model.stanc <- rstan::stanc(model.src)
  model.stanm <- rstan::stan_model(
    stanc_ret = model.stanc,
    model_name = 'prophet_model'
  )
})

save('model.stanm', file = model.binary)

packageStartupMessage('------ Model successfully compiled!')
packageStartupMessage('You can ignore any compiler warnings above.')
