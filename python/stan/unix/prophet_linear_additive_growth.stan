data {
  int T;                                // Sample size
  int<lower=1> K;                       // Number of seasonal vectors
  vector[T] t;                            // Day
  vector[T] y;                            // Time-series
  int S;                                // Number of changepoints
  matrix[T, S] A;                   // Split indicators
  real t_change[S];                 // Index of changepoints
  matrix[T,K] X;                // season vectors
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

  for (i in 1:S) {
    gamma[i] = -t_change[i] * delta[i];
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
  y ~ normal((k + A * delta) .* t + (m + A * gamma) + X * beta, sigma_obs);
}
