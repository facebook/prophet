// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

functions {
  real[ , ] get_changepoint_matrix(real[] t, real[] t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    real A[T, S];
    real a_row[S];
    int cp_idx;

    // Start with an empty matrix.
    A = rep_array(0, T, S);
    a_row = rep_array(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  real[] logistic_gamma(real k, real m, real[] delta, real[] t_change, int S) {
    real gamma[S];  // adjusted offsets, for piecewise continuity
    real k_s[S + 1];  // actual rate in each segment
    real m_pr;

    // Compute the rate in each segment
    k_s[1] = k;
    for (i in 1:S) {
      k_s[i + 1] = k_s[i] + delta[i];
    }

    // Piecewise offsets
    m_pr = m; // The offset in the previous segment
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // update for the next segment
    }
    return gamma;
  }
  
  real[] logistic_trend(
    real k,
    real m,
    real[] delta,
    real[] t,
    real[] cap,
    real[ , ] A,
    real[] t_change,
    int S,
    int T
  ) {
    real gamma[S];
    real Y[T];

    gamma = logistic_gamma(k, m, delta, t_change, S);
    for (i in 1:T) {
      Y[i] = cap[i] / (1 + exp(-(k + dot_product(A[i], delta))
        * (t[i] - (m + dot_product(A[i], gamma)))));
    }
    return Y;
  }

  // Linear trend function

  real[] linear_trend(
    real k,
    real m,
    real[] delta,
    real[] t,
    real[ , ] A,
    real[] t_change,
    int S,
    int T
  ) {
    real gamma[S];
    real Y[T];

    for (i in 1:S) {
      gamma[i] = -t_change[i] * delta[i];
    }
    for (i in 1:T) {
      Y[i] = (k + dot_product(A[i], delta)) * t[i] + (
        m + dot_product(A[i], gamma));
    }
    return Y;
  }

   // Flat trend function

    real[] flat_trend(
    real m,
    int T
  ) {
    return rep_array(m, T);
  }


}

data {
  int T;                // Number of time periods
  int<lower=1> K;       // Number of regressors
  real t[T];            // Time
  real cap[T];          // Capacities for logistic trend
  real y[T];            // Time series
  int S;                // Number of changepoints
  real t_change[S];     // Times of trend changepoints
  real X[T,K];         // Regressors
  vector[K] sigmas;     // Scale on seasonality prior
  real<lower=0> tau;    // Scale on changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic, 2 for flat
  real s_a[K];          // Indicator of additive features
  real s_m[K];          // Indicator of multiplicative features
}

transformed data {
  real A[T, S];
  A = get_changepoint_matrix(t, t_change, T, S);
}

parameters {
  real k;                   // Base trend growth rate
  real m;                   // Trend offset
  real delta[S];            // Trend rate adjustments
  real<lower=0> sigma_obs;  // Observation noise
  real beta[K];             // Regressor coefficients
}

transformed parameters {
  real trend[T];
  real Y[T];
  real beta_m[K];
  real beta_a[K];

  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change, S, T);
  } else if (trend_indicator == 1) {
    trend = logistic_trend(k, m, delta, t, cap, A, t_change, S, T);
  } else if (trend_indicator == 2){
    trend = flat_trend(m, T);
  }

  for (i in 1:K) {
    beta_m[i] = beta[i] * s_m[i];
    beta_a[i] = beta[i] * s_a[i];
  }

  for (i in 1:T) {
    Y[i] = (
      trend[i] * (1 + dot_product(X[i], beta_m)) + dot_product(X[i], beta_a)
    );
  }
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.5);
  beta ~ normal(0, sigmas);

  // Likelihood
  y ~ normal(Y, sigma_obs);
}
