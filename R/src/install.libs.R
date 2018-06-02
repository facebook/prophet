

packageStartupMessage('Compiling model (this will take a minute...)')

dest <- file.path(R_PACKAGE_DIR, paste0('libs', R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)

packageStartupMessage(paste('Writing model to:', dest))
packageStartupMessage(paste('Compiling using binary:', R.home('bin')))

model.src <- file.path(R_PACKAGE_SOURCE, 'inst', 'stan', 'prophet.stan')
model.binary <- file.path(dest, 'prophet_stan_model.RData')
model.stanc <- rstan::stanc(model.src)
model.stanm <- rstan::stan_model(
  stanc_ret = model.stanc,
  model_name = 'prophet_model'
)
save('model.stanm', file = model.binary)

packageStartupMessage('------ Model successfully compiled!')
packageStartupMessage('You can ignore any compiler warnings above.')
