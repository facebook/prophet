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
    return cap .* inv_logit((k + A * delta) .* (t - (m + A * gamma)));
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
  
  // Estimate quantiles from a series
  
  row_vector quantile(row_vector unordered_x, int[] q_probs) {
    int num_q_probs = num_elements(q_probs);
    int sample_size = num_elements(unordered_x);
    row_vector[sample_size] ordered_x = sort_asc(unordered_x);
    
    row_vector[num_q_probs] h = (sample_size - 1) * (to_row_vector(q_probs) / 100) + 1;  
    row_vector[num_q_probs] Q;
    
    for (quant_index in 1:num_q_probs) {
      int h_floor = (((sample_size - 1) * q_probs[quant_index]) / 100) + 1;  
      
      Q[quant_index] = ordered_x[h_floor] + (h[quant_index] - h_floor) * (ordered_x[h_floor + 1] - ordered_x[h_floor]);
    }
   
    return(Q); 
  }
  
  // predictions and  trend quantiles
  
  matrix get_prediction_quantiles_rng(
    real k, real m, vector delta, vector t_pred, vector cap_pred, 
    vector t_change, int S, int S_pred, int T_pred, int n_samp, int trend_indicator, 
    vector mul_seas_hat, vector add_seas_hat, real sigma_obs, int[] q_probs
  ) {
    
    real lambda;
    vector[S + S_pred] t_change_sim;
    vector[S + S_pred] delta_sim;
    matrix[T_pred, S + S_pred] A_sim;
    vector[T_pred] error_sim;
    matrix[T_pred, n_samp] trend_samples;
    matrix[T_pred, n_samp] yhat_samples;
    matrix[T_pred, 4] trend_and_yhat_quantiles;


    // Set the first S changepoints as fitted
    for (i in 1:S) {
      t_change_sim[i] = t_change[i];
      delta_sim[i] = delta[i];
    }
    // Get the Laplace scale
    lambda = mean(fabs(delta)) + 1e-8;

    // trend  and yhat Samples
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
      
      // Sample Trend
      trend_samples[:, i] = 
      get_trend(
        k, m, delta_sim, t_pred, cap_pred, A_sim, t_change_sim, S + S_pred, trend_indicator
        );
     
      // Sample Errors  
      for (j in 1:T_pred) { error_sim[j] = normal_rng(0, sigma_obs); }
      
      // Sample Predictions
      yhat_samples[:, i] = 
        trend_samples[:, i] .* (1 + mul_seas_hat) + add_seas_hat + error_sim;
    } 
    
    // Calculate quantiles for predictions and trend; and
    // store them in a 4 column matrix
    for (i in 1:T_pred) {
      trend_and_yhat_quantiles[i, 1:2] = quantile(yhat_samples[i, :], q_probs);
      trend_and_yhat_quantiles[i, 3:4] = quantile(trend_samples[i, :], q_probs);
    }
    
    return(trend_and_yhat_quantiles);
  }
  
} //functions

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

  int q_probs[2];
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
  sigma_obs ~ normal(0, 0.5);
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
  matrix[T_pred, 4] prediction_quantiles;
  
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
    prediction_quantiles = get_prediction_quantiles_rng(
      k, m, delta, t_pred, cap_pred, 
      t_change, S, S_pred, T_pred, n_samp, trend_indicator, 
      mul_seas_hat, add_seas_hat, sigma_obs, q_probs
    );
    
  }
}
