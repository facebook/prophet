data {
  int T;                          // sample size
  int<lower=1> K;                 // number of seasonal vectors
  vector[T] t;                    // day
  vector[T] y;                    // time-series
  int S;                          // number of changepoints
  matrix[T, S] A;                 // split indicators
  vector[S] t_change;             // index of changepoints
  matrix[T, K] X;                 // season vectors
  vector[K] sigmas;               // scale on seasonality prior
  real<lower=0> tau;              // scale on changepoints prior
}
parameters {
  real k;                         // base growth rate
  real m;                         // offset
  vector[S] delta;                // rate adjustments
  real<lower=0> sigma_obs;        // error scale (incl. seasonal variation)
  vector[K] beta;                 // seasonal vector
}
transformed parameters {
  // adjusted offsets for piecewise continuity
  vector[S] gamma = -t_change .* delta;
}

model {
  // priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.5);
  beta ~ normal(0, sigmas);

  // likelihood
  y ~ normal((k + A * delta) .* t + (m + A * gamma) + X * beta, sigma_obs);
}
