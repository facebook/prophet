

packageStartupMessage('Compiling models (this will take a minute...)')

dest <- file.path(R_PACKAGE_DIR, paste0('libs', R_ARCH))
dir.create(dest, recursive = TRUE, showWarnings = FALSE)

packageStartupMessage(paste('Writing models to:', dest))
packageStartupMessage(paste('Compiling using binary:', R.home('bin')))

logistic.growth.src <- file.path(R_PACKAGE_SOURCE, 'inst', 'stan', 'prophet_logistic_growth.stan')
logistic.growth.binary <- file.path(dest, 'prophet_logistic_growth.RData')
logistic.growth.stanc <- rstan::stanc(logistic.growth.src)
logistic.growth.stanm <- rstan::stan_model(stanc_ret = logistic.growth.stanc,
                                           model_name = 'logistic_growth')
save('logistic.growth.stanm', file = logistic.growth.binary)

linear.growth.src <- file.path(R_PACKAGE_SOURCE, 'inst', 'stan', 'prophet_linear_growth.stan')
linear.growth.binary <- file.path(dest, 'prophet_linear_growth.RData')
linear.growth.stanc <- rstan::stanc(linear.growth.src)
linear.growth.stanm <- rstan::stan_model(stanc_ret = linear.growth.stanc,
                                         model_name = 'linear_growth')
save('linear.growth.stanm', file = linear.growth.binary)

packageStartupMessage('------ Models successfully compiled!')
packageStartupMessage('You can ignore any compiler warnings above.')
