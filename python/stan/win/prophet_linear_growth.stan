data {
  int T;                                // Sample size
  int<lower=1> K;                       // Number of seasonal vectors
  real t[T];                            // Day
  real y[T];                            // Time-series
  int S;                                // Number of changepoints
  real A[T, S];                   // Split indicators
  real t_change[S];                 // Index of changepoints
  real X[T,K];                // season vectors
  vector[K] sigmas;              // scale on seasonality prior
  real<lower=0> tau;                  // scale on changepoints prior
}

parameters {
  real k;                            // Base growth rate
  real m;                            // offset
  real delta[S];                       // Rate adjustments
  real<lower=0> sigma_obs;               // Observation noise (incl. seasonal variation)
  real beta[K];                    // seasonal vector
}

transformed parameters {
  real gamma[S];                  // adjusted offsets, for piecewise continuity

  for (i in 1:S) {
    gamma[i] = -t_change[i] * delta[i];
  }
}

model {
  real Y[T];

  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.5);
  beta ~ normal(0, sigmas);

  // Likelihood
  for (i in 1:T) {
    Y[i] = (dot_product(A[i], delta) + k) * t[i] + (dot_product(A[i], gamma) + m) + dot_product(X[i], beta);
  }
  y ~ normal(Y, sigma_obs);
}
