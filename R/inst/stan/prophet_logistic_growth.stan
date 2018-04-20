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
}

data {
  int T;                                // Sample size
  int<lower=1> K;                       // Number of seasonal vectors
  vector[T] t;                            // Day
  vector[T] cap;                          // Capacities
  vector[T] y;                            // Time-series
  int S;                                // Number of changepoints
  vector[S] t_change;                 // Index of changepoints
  matrix[T,K] X;                    // season vectors
  vector[K] sigmas;               // scale on seasonality prior
  real<lower=0> tau;                  // scale on changepoints prior
}

transformed data {
  matrix[T, S] A;
  A = get_changepoint_matrix(t, t_change, T, S);
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
    gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
    m_pr = m_pr + gamma[i];  // update for the next segment
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
  y ~ normal(cap ./ (1 + exp(-(k + A * delta) .* (t - (m + A * gamma)))) + X * beta, sigma_obs);
}
