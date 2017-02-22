# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant 
# of patent rights can be found in the PATENTS file in the same directory.

data {
  int T;                                // Sample size
  int<lower=1> K;                       // Number of seasonal vectors
  vector[T] t;                            // Day
  vector[T] y;                            // Time-series
  int S;                                // Number of split points
  matrix[T, S] A;                   // Split indicators
  int s_indx[S];                 // Index of split points
  matrix[T,K] X;                // season vectors
  real<lower=0> sigma;              // scale on seasonality prior
  real<lower=0> tau;                  // scale on changepoints prior
}

parameters {
  real k;                            // Base growth rate
  real m;                            // offset
  vector[S] delta;                       // Rate adjustments
  real<lower=0> sigma_obs;               // Observation noise (incl. seasonal variation)
  vector[K] beta;                    // seasonal vector
}

transformed parameters {
  vector[S] gamma;                  // adjusted offsets, for piecewise continuity

  for (i in 1:S) {
    gamma[i] = -t[s_indx[i]] * delta[i];
  }
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.5);
  beta ~ normal(0, sigma);

  // Likelihood
  y ~ normal((k + A * delta) .* t + (m + A * gamma) + X * beta, sigma_obs);
}
