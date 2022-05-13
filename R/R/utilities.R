# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#' Summarise the coefficients of the extra regressors used in the model.
#' For additive regressors, the coefficient represents the incremental impact
#' on \code{y} of a unit increase in the regressor. For multiplicative regressors,
#' the incremental impact is equal to \code{trend(t)} multiplied by the coefficient.
#'
#' Coefficients are measured on the original scale of the training data.
#'
#' @param m Prophet model object, after fitting.
#'
#' @return Dataframe with one row per regressor.
#' @details Output dataframe columns:
#' \itemize{
#'   \item{regressor: Name of the regressor}
#'   \item{regressor_mode: Whether the regressor has an additive or multiplicative
#' effect on \code{y}.}
#'   \item{center: The mean of the regressor if it was standardized. Otherwise 0.}
#'   \item{coef_lower: Lower bound for the coefficient, estimated from the MCMC samples.
#'     Only different to \code{coef} if \code{mcmc_samples > 0}.
#'   }
#'   \item{coef: Expected value of the coefficient.}
#'   \item{coef_upper: Upper bound for the coefficient, estimated from MCMC samples.
#'     Only to different to \code{coef} if \code{mcmc_samples > 0}.
#'   }
#' }
#'
#' @export
regressor_coefficients <- function(m){
  if (length(m$extra_regressors) == 0) {
    stop("No extra regressors found.")
  }
  regr_names <- names(m$extra_regressors)
  regr_modes <- unlist(lapply(m$extra_regressors, function(x) x$mode))
  regr_mus <- unlist(lapply(m$extra_regressors, function (x) x$mu))
  regr_stds <- unlist(lapply(m$extra_regressors, function(x) x$std))

  beta_indices <- which(m$train.component.cols[, regr_names] == 1, arr.ind = TRUE)[, "row"]
  betas <- m$params$beta[, beta_indices, drop = FALSE]
  # If regressor is additive, multiply by the scale factor to put coefficients on the original training data scale.
  y_scale_indicator <- matrix(
    data = ifelse(regr_modes == "additive", m$y.scale, 1),
    nrow = nrow(betas),
    ncol = ncol(betas),
    byrow = TRUE
  )
  coefs <- betas * y_scale_indicator  / regr_stds

  percentiles = c((1 - m$interval.width) / 2, 1 - (1 - m$interval.width) / 2)
  bounds <- apply(betas, 2, stats::quantile, probs = percentiles)

  df <- data.frame(
    regressor = regr_names,
    regressor_mode = regr_modes,
    center = regr_mus,
    coef_lower = bounds[1, ],
    coef = apply(betas, 2, mean),
    coef_upper = bounds[2, ],
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  return(df)
}
