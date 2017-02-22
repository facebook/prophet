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
  vector[T] cap;                          // Capacities
  vector[T] y;                            // Time-series
  int S;                                // Number of split points
  matrix[T, S] A;                   // Split indicators
  int s_indx[S];                 // Index of split points
  matrix[T,K] X;                    // season vectors
  real<lower=0> sigma;              // scale on seasonality prior
  real<lower=0> tau;                  // scale on changepoints prior
}


transformed data {
  int s_ext[S + 1];  // Segment endpoints
  for (j in 1:S) {
    s_ext[j] = s_indx[j];
  }
  s_ext[S + 1] = T + 1;  // Used for the m_adj loop below.
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
  vector[S + 1] k_s;                 // actual rate in each segment
  real m_pr;

  // Compute the rate in each segment
  k_s[1] = k;
  for (i in 1:S) {
    k_s[i + 1] = k_s[i] + delta[i];
  }

  // Piecewise offsets
  m_pr = m; // The offset in the previous segment
  for (i in 1:S) {
    gamma[i] = (t[s_indx[i]] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
    m_pr = m_pr + gamma[i];  // update for the next segment
  }
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.1);
  beta ~ normal(0, sigma);

  // Likelihood
  y ~ normal(cap ./ (1 + exp(-(k + A * delta) .* (t - (m + A * gamma)))) + X * beta, sigma_obs);
}
