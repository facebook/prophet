#' Get the stan backend defined in the environment variables.
#'
#' @return 'rstan' or 'cmdstanr'. 'rstan' if variable is not set.
#' @keywords internal
get_stan_backend <- function() {
  backend_setting <- Sys.getenv("R_STAN_BACKEND", "RSTAN")
  if (backend_setting %in% c("RSTAN", "CMDSTANR")) {
    backend <- switch(
      backend_setting,
      "RSTAN" = "rstan",
      "CMDSTANR" = "cmdstanr"
    )
    if (backend == "cmdstanr") check_cmdstanr()
    return(backend)
  } else {
    return("rstan")
  }
}

#' Check that the required packages for using the cmdstanr backend are installed.
#'
#' @return NULL if successful, and prints the current version of cmdstan being used.
#' @keywords internal
check_cmdstanr <- function() {
  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop(
      "Package \"cmdstanr\" needed to use cmdstanr backend. See installation instructions: https://mc-stan.org/cmdstanr/.",
      call. = FALSE
    )
  }
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop(
      "Package \"posterior\" needed to use cmdstanr backend. See installation instructions: https://mc-stan.org/posterior/.",
      call. = FALSE
    )
  }
  cmdstanr_version <- cmdstanr::cmdstan_version()
  return(invisible(TRUE))
}

#' Load the Prophet Stan model.
#'
#' @param backend "rstan" or "cmdstanr".
#'
#' @return stanmodel object if backend = "rstan", CmdStanModel object if backend = "cmdstanr"
#' @keywords internal
.load_model <- function(backend) {
  switch(
    backend,
    "rstan" = .load_model_rstan(),
    "cmdstanr" = .load_model_cmdstanr()
  )
}

#' @rdname .load_model
.load_model_rstan <- function() {
  if (exists(".prophet.stan.model", where = prophet_model_env)) {
    model <- get('.prophet.stan.model', envir = prophet_model_env)
  } else {
    model <- stanmodels$prophet
  }

  return(model)
}

#' @rdname .load_model
.load_model_cmdstanr <- function() {
  model_file <- system.file(
    "stan",
    "prophet.stan",
    package = "prophet",
    mustWork = TRUE
  )
  model <- cmdstanr::cmdstan_model(model_file)

  return(model)
}

#' Gives Stan arguments the appropriate names depending on the chosen Stan backend.
#'
#' @param model Model object.
#' @param dat List containing data to use in fitting.
#' @param stan_init Function to initialize parameters for stan fit.
#' @param backend "rstan" or "cmdstanr".
#' @param type "mcmc" or "optimize".
#' @param mcmc_samples Integer, if greater than 0, will do full Bayesian
#'  inference with the specified number of MCMC samples. If 0, will do MAP
#'  estimation.
#'
#' @return Named list of arguments.
#' @keywords internal
.stan_args <- function(model, dat, stan_init, backend, type, mcmc_samples = 0, ...) {
  args <- switch(
    backend,
    "rstan" = .stan_args_rstan(model, dat, stan_init, type, mcmc_samples),
    "cmdstanr" = .stan_args_cmdstanr(model, dat, stan_init, type, mcmc_samples)
  )
  args <- utils::modifyList(args, list(...))

  return(args)
}

#' @rdname .stan_args
.stan_args_rstan <- function(model, dat, stan_init, type, mcmc_samples = NULL) {
  if (type == "mcmc") {
    args <- list(
      object = model,
      data = dat,
      init = stan_init,
      iter = mcmc_samples,
      chains = 4
    )
  } else if (type == "optimize") {
    args <- list(
      object = model,
      data = dat,
      init = stan_init,
      algorithm = if(dat$T < 100) {'Newton'} else {'LBFGS'},
      iter = 1e4,
      as_vector = FALSE
    )
  }

  return(args)
}

#' @rdname .stan_args
.stan_args_cmdstanr <- function(model, dat, stan_init, type, mcmc_samples = NULL) {
  if (type == "mcmc") {
    args <- list(
      object = model,
      data = dat,
      init = stan_init,
      iter_warmup = mcmc_samples / 2,
      iter_sampling = mcmc_samples / 2,
      chains = 4,
      refresh = 0,
      show_messages = FALSE
    )
  } else if (type == "optimize") {
    args <- list(
      object = model,
      data = dat,
      init = stan_init,
      algorithm = if(dat$T < 100) {'newton'} else {'lbfgs'},
      iter = 1e4,
      refresh = 0
    )
  }

  return(args)
}

#' Obtain the point estimates of the parameters of the Prophet model using
#' stan's optimization algorithms.
#'
#' @param args Named list of arguments suitable for the chosen backend. Must
#'   include arguments required for optimization.
#' @param backend "rstan" or "cmdstanr".
#'
#' @return A named list containing "stan_fit" (the fitted stan object),
#'   "params", and "n_iteration"
#' @keywords internal
.fit <- function(args, backend) {
  switch(
    backend,
    "rstan" = .fit_rstan(args),
    "cmdstanr" = .fit_cmdstanr(args)
  )
}

#' Obtain the joint posterior distribution of the parameters of the Prophet
#' model using MCMC sampling.
#'
#' @param args Named list of arguments suitable for the chosen backend. Must
#'   include arguments required for MCMC sampling.
#' @param backend "rstan" or "cmdstanr".
#'
#' @return A named list containing "stan_fit" (the fitted stan object),
#'   "params", and "n_iteration"
#' @keywords internal
.sampling <- function(args, backend) {
  switch(
    backend,
    "rstan" = .sampling_rstan(args),
    "cmdstanr" = .sampling_cmdstanr(args)
  )
}

#' @rdname .fit
.fit_rstan <- function(args) {
  model_output <- list()
  model_output$stan_fit <- do.call(rstan::optimizing, args)
  if (model_output$stan_fit$return_code != 0) {
    message(
      'Optimization terminated abnormally. Falling back to Newton optimizer.'
    )
    args$algorithm = 'Newton'
    model_output$stan_fit <- do.call(rstan::optimizing, args)
  }
  model_output$params <- model_output$stan_fit$par
  model_output$n_iteration <- 1

  return(model_output)
}

#' @rdname .sampling
.sampling_rstan <- function(args) {
  model_output <- list()
  model_output$stan_fit <- do.call(rstan::sampling, args)
  model_output$params <- rstan::extract(model_output$stan_fit)
  model_output$n_iteration <- length(model_output$params$k)

  return(model_output)
}

#' @rdname .fit
.fit_cmdstanr <- function(args) {
  # TODO: Replace with method to extract parameter names once implemented in cmdstanr
  param_names <- c("k", "m", "delta", "sigma_obs", "beta", "trend")
  model_output <- list()
  model <- args$object
  args$object <- NULL
  model_output$stan_fit <- do.call(model$optimize, args)
  if (model_output$stan_fit$return_codes()[1] != 0) {
    message(
      'Optimization terminated abnormally. Falling back to Newton optimizer.'
    )
    args$algorithm = 'newton'
    model_output$stan_fit <- do.call(model$optimize, args)
  }
  model_output$params <- list()
  for (name in param_names) {
    model_output$params[[name]] <- unname(model_output$stan_fit$mle(name))
  }
  model_output$n_iteration <- 1

  return(model_output)
}

#' @rdname .sampling
.sampling_cmdstanr <- function(args) {
  param_names <- c("k", "m", "delta", "sigma_obs", "beta", "trend")
  model_output <- list()
  model <- args$object
  args$object <- NULL
  param_names <- c(param_names, "lp__")
  model_output$stan_fit <- do.call(model$sample, args)
  model_output$params <- list()
  for (name in param_names) {
    model_output$params[[name]] <- posterior::as_draws_matrix(model_output$stan_fit$draws(name))
  }
  model_output$n_iteration <- nrow(model_output$params$k)

  return(model_output)
}
