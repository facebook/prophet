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
        cp_idx += 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  vector logistic_gamma(real k, real m, vector delta, vector t_change, int S) {
    vector[S] gamma;  // adjusted offsets, for piecewise continuity
    vector[S + 1] k_s;  // actual rate in each segment
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
  
  vector logistic_trend(
    real k,
    real m,
    vector delta,
    vector t,
    vector cap,
    matrix A,
    vector t_change,
    int S,
    int T
  ) {
    vector[S] gamma;
    vector[T] Y;

    gamma = logistic_gamma(k, m, delta, t_change, S);
    for (i in 1:T) {
      Y[i] = cap[i] / (1 + exp(-(k + dot_product(A[i], delta)) * (t[i] - (m + dot_product(A[i], gamma)))))
    }
    return Y;
  }

  // Linear trend function

  vector linear_trend(
    real k,
    real m,
    vector delta,
    vector t,
    matrix A,
    vector t_change,
    int S,
    int T
  ) {
    vector[S] gamma;
    vector[T] Y;
    
    gamma = (-t_change .* delta);
    for (i in 1:T) {
      Y[i] = (k + dot_product(A[i], delta)) * t[i] + (m + dot_product(A[i], gamma))
    }
    return Y;
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
}

transformed data {
  matrix[T, S] A;
  A = get_changepoint_matrix(t, t_change, T, S);
}

parameters {
  real k;                   // Base trend growth rate
  real m;                   // Trend offset
  vector[S] delta;          // Trend rate adjustments
  real<lower=0> sigma_obs;  // Observation noise
  vector[K] beta;           // Regressor coefficients
}

transformed parameters {
  vector[T] trend;
  vector[T] Y;

  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change, S, T);
  } else if (trend_indicator == 1) {
    trend = logistic_trend(k, m, delta, t, cap, A, t_change, S, T);
  }

  for (i in 1:T) {
    Y[i] = trend[i] + dot_product(X[i], beta);
  }
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.1);
  beta ~ normal(0, sigmas);

  // Likelihood
  y ~ normal(Y, sigma_obs);
}
