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
    real k, real m, vector delta, vector t, vector cap, matrix A,
    vector t_change, int S
  ) {
    vector[S] gamma;

    gamma = logistic_gamma(k, m, delta, t_change, S);
    return cap ./ (1 + exp(-(k + A * delta) .* (t - (m + A * gamma))));
  }

  // Linear trend function

  vector linear_trend(
    real k, real m, vector delta, vector t, matrix A, vector t_change
  ) {
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }
  
  // Helper for getting appropriate trend
  
  vector get_trend(
    real k, real m, vector delta, vector t, vector cap, matrix A,
    vector t_change, int S, int trend_indicator
  ) {
    if (trend_indicator == 0) {
      return linear_trend(k, m, delta, t, A, t_change);
    } else {
      return logistic_trend(k, m, delta, t, cap, A, t_change, S);
    }
  }
}

data {
  int T;                    // Number of time periods
  int<lower=1> K;           // Number of regressors
  vector[T] t;              // Time
  vector[T] cap;            // Capacities for logistic trend
  vector[T] y;              // Time series
  int S;                    // Number of changepoints
  vector[S] t_change;       // Times of trend changepoints
  matrix[T, K] X;           // Regressors
  vector[K] sigmas;         // Scale on seasonality prior
  real<lower=0> tau;        // Scale on changepoints prior
  int trend_indicator;      // 0 for linear, 1 for logistic
  vector[K] s_a;            // Indicator of additive features
  vector[K] s_m;            // Indicator of multiplicative features

  int T_pred;               // Number of prediction time periods
  vector[T_pred] t_pred;    // Times for predictions
  vector[T_pred] cap_pred;  // Predictive capacities
  matrix[T_pred, K] X_pred; // Predictive features
  int n_samp;               // Number of samples for trend change uncertainty
  int S_pred;               // Upper bound on number of future changepoints
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

model {
  // Priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.1);
  beta ~ normal(0, sigmas);

   // Likelihood
   y ~ normal(
     get_trend(k, m, delta, t, cap, A, t_change, S, trend_indicator)
      .* (1 + X * (beta .* s_m))
      + X * (beta .* s_a),
      sigma_obs
   );
  
}

generated quantities {
  // Make predictions.
  vector[T_pred] y_hat;
  vector[T_pred] mul_seas_hat;
  vector[T_pred] add_seas_hat;
  vector[T_pred] trend_hat;
  matrix[T_pred, S] A_pred;
  matrix[T_pred, n_samp] trend_samples;
  matrix[T_pred, n_samp] yhat_samples;
  vector[S + S_pred] t_change_sim;
  vector[S + S_pred] delta_sim;
  real lambda;
  matrix[T_pred, S + S_pred] A_sim;
  vector[T_pred] error_sim;

  if (T_pred > 0) {
    // Get the main estimate
    mul_seas_hat = X_pred * (beta .* s_m);
    add_seas_hat = X_pred * (beta .* s_a);

    A_pred = get_changepoint_matrix(t_pred, t_change, T_pred, S);
    trend_hat = get_trend(
      k, m, delta, t_pred, cap_pred, A_pred, t_change, S, trend_indicator
    );
    y_hat = trend_hat .* (1 + mul_seas_hat) + add_seas_hat;

    // Estimate uncertainty with the generative model

    // Set the first S changepoints as fitted
    for (i in 1:S) {
      t_change_sim[i] = t_change[i];
      delta_sim[i] = delta[i];
    }
    // Get the Laplace scale
    lambda = mean(fabs(delta)) + 1e-8;

    for (i in 1:n_samp) {
      if (S_pred > 0) {
        // Sample new changepoints from a Poisson process with rate S
        // Sample changepoint deltas from Laplace(lambda)
        t_change_sim[S + 1] = 1 + exponential_rng(S);
        for (j in (S + 2):(S + S_pred)) {
          t_change_sim[j] = t_change_sim[j - 1] + exponential_rng(S);
        }
        for (j in (S + 1):(S + S_pred)) {
          delta_sim[j] = double_exponential_rng(0, lambda);
        }
      }
      // Compute trend with these changepoints
      A_sim = get_changepoint_matrix(t_pred, t_change_sim, T_pred, S + S_pred);
      trend_samples[:, i] = 
      get_trend(
        k, m, delta_sim, t_pred, cap_pred, A_sim, t_change_sim, S + S_pred, trend_indicator
        );
      
      # Sample Errors  
      for (j in 1:T_pred) {
        error_sim[j] = normal_rng(0, sigma_obs);
      }    
      yhat_samples[:, i] = trend_samples[:, i] .* (1 + mul_seas_hat) 
                           + add_seas_hat 
                           + error_sim;
      
    }
  }
}
