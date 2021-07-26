from jax import jit
from jax.lax import scan
import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import sample, plate, deterministic
from numpyro.infer import SVI
from numpyro.infer.autoguide import AutoDelta, Trace_ELBO
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Minimize
from jax import random


def construct_changepoint_matrix(t, t_change, T, S):
    A = np.zeros((T, S))
    a_row = np.zeros(S)
    cp_idx = 1 if S > 1 else 0
    for i in range(T):
        while (cp_idx < S) and (t[i] >= t_change[cp_idx]):
            a_row[cp_idx] = 1
            cp_idx += 1
        A[i] = a_row
    return jnp.array(A)


@jit
def linear_trend(k: float, m: float, delta: jnp.ndarray, t: jnp.ndarray, A: jnp.ndarray, t_change: jnp.ndarray):
    first_term = jnp.multiply((k + A.dot(delta)), t)
    second_term = (m + A.dot(jnp.multiply(-t_change, delta)))
    return first_term + second_term


def transition_function(carry, index):
    m_pr, t_change, k_s = carry
    gamma_i = (t_change[index] - m_pr) * (1 - k_s[index] / k_s[index + 1])
    return (m_pr + gamma_i, t_change, k_s), gamma_i


def logistic_gamma(k, m, delta, t_change, S):
    k_s = jnp.append(jnp.array(k), k + jnp.cumsum(delta))
    _, gamma = scan(transition_function, (m, t_change, k_s), jnp.arange(S))
    return gamma


def logistic_trend(k, m, delta, t, cap, A, t_change, S):
    gamma = logistic_gamma(k, m, delta, t_change, S)
    inv_logit = expit(jnp.multiply(k + A.dot(delta), (t - (m + A.dot(gamma)))))
    return jnp.multiply(cap, inv_logit)


@jit
def compute_mu(trend, X, betas, s_m, s_a):
    return jnp.multiply(trend, (1 + X.dot(betas * s_m))) + X.dot(betas * s_a)


def model_linear(s_m, s_a, sigmas, X, cap, t, t_change, tau, T, S, y, changepoint):
    k = sample("k", dist.Normal(0, 5))
    m = sample("m", dist.Normal(0, 5))
    with plate('delta_dim', S):
        delta = sample("delta", dist.Laplace(0, tau))
        trend = deterministic("trend", linear_trend(k, m, delta, t, changepoint, t_change))
    sigma_obs = sample("sigma_obs", dist.HalfNormal(scale=0.5))
    betas = sample("beta", dist.MultivariateNormal(0, jnp.diag(sigmas)))
    mu = deterministic("mu", compute_mu(trend, X, betas, s_m, s_a))
    with plate("data", T):
        sample("obs", dist.Normal(mu, sigma_obs), obs=y)


def model_logistic(s_m, s_a, sigmas, X, cap, t, t_change, tau, T, S, y, changepoint):
    k = sample("k", dist.Normal(0, 5))
    m = sample("m", dist.Normal(0, 5))
    with plate('delta_dim', S):
        delta = sample("delta", dist.Laplace(0, tau))
        trend = deterministic("trend", logistic_trend(k, m, delta, t, cap, changepoint, t_change, S))
    sigma_obs = sample("sigma_obs", dist.HalfNormal(scale=0.5))
    betas = sample("beta", dist.MultivariateNormal(0, jnp.diag(sigmas)))
    mu = deterministic("mu", compute_mu(trend, X, betas, s_m, s_a))
    with plate("data", T):
        sample("obs", dist.Normal(mu, sigma_obs), obs=y)


def _dat_to_jax_arrays(dat):
    y = jnp.array(dat['y'])
    X = jnp.array(dat['X'])
    t = jnp.array(dat['t'])
    s_m = jnp.array(dat['s_m'])
    s_a = jnp.array(dat['s_a'])
    sigmas = jnp.array(dat['sigmas'])
    t_change = jnp.array(dat['t_change'])
    cap = jnp.array(dat['cap'])
    return y, X, t, s_m, s_a, sigmas, t_change, cap


def run_mcmc(dat, n_iterations):
    changepoint = construct_changepoint_matrix(dat['t'], dat['t_change'], dat['T'], dat['S'])

    if dat['trend_indicator'] == 0:
        nuts_kernel = NUTS(model_linear)
    else:
        nuts_kernel = NUTS(model_logistic)

    mcmc = MCMC(nuts_kernel, num_warmup=n_iterations, num_samples=n_iterations)

    rng_key = random.PRNGKey(0)
    y, X, t, s_m, s_a, sigmas, t_change, cap = _dat_to_jax_arrays(dat)

    mcmc.run(rng_key, s_m, s_a, sigmas, X, cap, t, t_change, dat['tau'], dat['T'], dat['S'], y, changepoint)
    samples = mcmc.get_samples()
    return {k: np.array(samples[k]) for k in samples}


def find_map(dat, init_args):
    changepoint = construct_changepoint_matrix(dat['t'], dat['t_change'], dat['T'], dat['S'])
    y, X, t, s_m, s_a, sigmas, t_change, cap = _dat_to_jax_arrays(dat)

    optimizer = Minimize(method="BFGS")

    init_fn = init_to_value(values={k: jnp.array(v) for k, v in init_args.items()})

    if dat['trend_indicator'] == 0:
        guide = AutoDelta(model_linear, init_loc_fn=init_fn)
        model_fn = model_linear
    else:
        guide = AutoDelta(model_logistic, init_loc_fn=init_fn)
        model_fn = model_logistic

    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    init_state = svi.init(random.PRNGKey(0), s_m, s_a, sigmas, X, cap, t,
                          t_change, dat['tau'], dat['T'], dat['S'], y, changepoint)
    optimal_state, loss = svi.update(init_state, s_m, s_a, sigmas, X, cap, t,
                                     t_change, dat['tau'], dat['T'], dat['S'], y, changepoint)
    params = svi.get_params(optimal_state)
    params = {k.replace("_auto_loc", ""): np.array(v).reshape((1, -1)) for k, v in params.items()}
    return params

