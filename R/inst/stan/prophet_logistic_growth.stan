data {
  int T;                             // sample size
  int<lower=1> K;                    // number of seasonal vectors
  vector[T] t;                       // day
  vector[T] cap;                     // capacities
  vector[T] y;                       // observed time series
  int S;                             // number of changepoints
  matrix[T, S] A;                    // split indicators
  vector[S] t_change;                // index of changepoints
  matrix[T, K] X;                    // season vectors
  vector[K] sigmas;                  // seasonality prior scale
  real<lower=0> tau;                 // changepoints prior scale
}
parameters {
  real k;                            // base growth rate
  real m;                            // offset in previous segment
  vector[S] delta;                   // rate adjustments
  real<lower=0> sigma_obs;           // error scale (incl. seasonal variation)
  vector[K] beta;                    // seasonal vector
}
transformed parameters {
  // rate in each segment
  vector[S + 1] k_s = append_row(k, k + cumulative_sum(delta));

  // adjusted offsets for piecewise continuity
  vector[S] gamma;
  real m_pr = m;
  for (i in 1:S) {
    gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
    m_pr = m_pr + gamma[i];  // update for the next segment
  }
}
model {
  // priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 10);
  beta ~ normal(0, sigmas);

  // likelihood
  y ~ normal(cap ./ inv_logit((k + A * delta) .* (t - (m + A * gamma)))
             + X * beta, sigma_obs);
}
