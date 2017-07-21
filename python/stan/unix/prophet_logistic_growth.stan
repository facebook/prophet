data {
  int T;                                // Sample size
  int<lower=1> K;                       // Number of seasonal vectors
  vector[T] t;                            // Day
  vector[T] cap;                          // Capacities
  vector[T] y;                            // Time-series
  int S;                                // Number of changepoints
  matrix[T, S] A;                   // Split indicators
  real t_change[S];                 // Index of changepoints
  matrix[T,K] X;                    // season vectors
  vector[K] sigmas;              // scale on seasonality prior
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
