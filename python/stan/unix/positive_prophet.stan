// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

functions {
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    matrix[T, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
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

  // Linear trend function

  vector linear_trend(
    real k,
    real m,
    vector delta,
    vector t,
    matrix A,
    vector t_change
  ) {
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }
}

data {
  int T;                // Number of time periods
  int<lower=1> K;       // Number of regressors
  vector[T] t;          // Time
  vector[T] cap;        // Capacities for logistic trend
  vector[T] y;          // Time series
  int S;                // Number of changepoints
  vector[S] t_change;   // Times of trend changepoints
  matrix[T,K] X;        // Regressors
  vector[K] sigmas;     // Scale on seasonality prior
  real<lower=0> tau;    // Scale on changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic
  vector[K] s_a;        // Indicator of additive features
  vector[K] s_m;        // Indicator of multiplicative features
}

transformed data {
  matrix[T, S] A;
  A = get_changepoint_matrix(t, t_change, T, S);
}

parameters {
  real k_a;                   // Base trend growth rate
  real m_a;                   // Trend offset
  real k_b;                   // Base trend growth rate
  real m_b;                   // Trend offset
  vector[S] delta;            // Trend rate adjustments
  vector[K] alpha;             // Regressor coefficients
  vector[K] beta;             // Regressor coefficients
}

model {
  //priors
  k_a ~ normal(0, 5);
  m_a ~ normal(0, 5);
  k_b ~ normal(0, 5);
  m_b ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  alpha ~ normal(0, sigmas);
  beta ~ normal(0, sigmas);

  // Likelihood
  y ~ gamma(
      log(1 + exp(linear_trend(k_a, m_a, delta, t, A, t_change)
      .* (1 + X * (alpha .* s_m))
      + X * (alpha .* s_a))),
      log(1 + exp(linear_trend(k_b, m_b, delta, t, A, t_change)
      .* (1 + X * (beta .* s_m))
      + X * (beta .* s_a)))
  );
}