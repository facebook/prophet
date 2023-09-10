from dataclasses import dataclass
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.special import expit
from jax.lax import scan
from numpyro import deterministic, sample


def linear_trend(
    k: float,
    m: float,
    delta: jnp.ndarray,
    t: jnp.ndarray,
    cap: jnp.ndarray,
    t_change: jnp.ndarray,
    A: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the trend value at each time point t assuming linear growth. The trend function g(t) is defined
    in equation 4 on page 10 of the original Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf

    Returns
    -------
    An array of trend values at each time point.
    """
    time_dependent = jnp.multiply((k + A.dot(delta)), t)
    constant = m + A.dot(jnp.multiply(-t_change, delta))
    return time_dependent + constant


def transition_function(
    carry: Tuple[float, np.ndarray, np.ndarray], index: int
) -> Tuple[Tuple[float, np.ndarray, np.ndarray], float]:
    """
    Helper function for calculating the offset parameter adjustments required at each changepoint.

    Parameters
    ----------
    carry: A tuple containing the previous offset parameter value, the changepoint times,
        and the cumulative growth rate after each changepoint.
    index: The index of the current changepoint.

    Returns
    -------
    A tuple containing:
        - A tuple containing the updated offset parameter value, the changepoint times, and the cumulative growth rate at each changepoint.
        - The offset parameter adjustment required at the current changepoint.
    """
    m_pr, t_change, k_s = carry
    gamma_i = (t_change[index] - m_pr) * (1 - k_s[index] / k_s[index + 1])
    return (m_pr + gamma_i, t_change, k_s), gamma_i


def logistic_gamma(
    k: float, m: float, delta: jnp.ndarray, t_change: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the offset parameter adjustments (gamma_j) required at each changepoint assuming logistic growth.
    The equation for gamma_j is defined in page 9 of the original Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf.

    Returns
    -------
    An array of offset parameter adjustments required at each changepoint.
    """
    k_s = jnp.append(jnp.array(k), k + jnp.cumsum(delta))
    _, gamma = scan(
        transition_function, (m, t_change, k_s), jnp.arange(t_change.shape[0])
    )
    return gamma


def logistic_trend(
    k: float,
    m: float,
    delta: jnp.ndarray,
    t: jnp.ndarray,
    cap: jnp.ndarray,
    t_change: jnp.ndarray,
    A: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the trend value at each time point t assuming logistic growth.
    The trend function g(t) is defined in equation 3 on page 9 of the original Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf.

    Returns
    -------
    An array of trend values at each time point.
    """
    gamma = logistic_gamma(k, m, delta, t_change)
    return cap * expit((k + jnp.matmul(A, delta)) * (t - (m + jnp.matmul(A, gamma))))


def flat_trend(
    k: float,
    m: float,
    delta: jnp.ndarray,
    t: jnp.ndarray,
    cap: jnp.ndarray,
    t_change: jnp.ndarray,
    A: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the trend value at each time point t assuming zero base growth and no changepoints.

    Returns
    -------
    An array of trend values at each time point.
    """
    return jnp.repeat(m, t.shape[0])


def construct_changepoint_matrix(t: np.ndarray, t_change: np.ndarray) -> jnp.ndarray:
    """
    Constructs the A(t) indicator matrix for changepoints as defined in page 9 of the original
    Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf

    Parameters
    ----------
    t: An array of normalized time values.
    t_change: The normalized time values at which changepoints are assumed to occur.

    Returns
    -------
    A matrix of shape (number of time points, number of changepoints). For each row (i.e. each time point),
    the value of column j is 1 if changepoint j has taken effect by that time point, and 0 otherwise.
    """
    T = t.shape[0]
    S = t_change.shape[0]
    A = np.zeros((T, S))
    a_row = np.zeros(S)
    cp_idx = 1 if S > 1 else 0
    for i in range(T):
        while (cp_idx < S) and (t[i] >= t_change[cp_idx]):
            a_row[cp_idx] = 1
            cp_idx += 1
        A[i] = a_row
    return jnp.array(A)


def compute_mu(
    trend: jnp.ndarray,
    X: jnp.ndarray,
    betas: jnp.ndarray,
    s_m: jnp.ndarray,
    s_a: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculates the expected value of the response variable y at each time point t.

    Parameters
    ----------
    trend: The trend value at each time point t.
    X: The matrix containing fourier values representing seasonality, holidays indicators, and exogenous regressor values.
    betas: The coefficients for the features in X.
    s_m: An indicator vector for the features in X that have a multiplicative effect on y.
    s_a: An indicator vector for the features in X that have an additive effect on y.
    """
    return jnp.multiply(trend, (1 + X.dot(betas * s_m))) + X.dot(betas * s_a)


def get_model(trend_indicator: int) -> Callable:
    trend_function: Callable
    if trend_indicator == 0:
        trend_function = linear_trend
    elif trend_indicator == 1:
        trend_function = logistic_trend
    elif trend_indicator == 2:
        trend_function = flat_trend
    else:
        raise ValueError(f"Invalid trend_indicator value: {trend_indicator}")

    def model(
        t: jnp.ndarray,
        y: jnp.ndarray,
        cap: jnp.ndarray,
        t_change: jnp.ndarray,
        A: jnp.ndarray,
        tau: float,
        X: jnp.ndarray,
        sigmas: jnp.ndarray,
        s_a: jnp.ndarray,
        s_m: jnp.ndarray,
    ) -> None:
        S = t_change.shape[0]
        k = sample("k", dist.Normal(0, 5))
        m = sample("m", dist.Normal(0, 5))
        delta = sample("delta", dist.Laplace(0, jnp.repeat(tau, S)))
        trend = deterministic("trend", trend_function(k, m, delta, t, cap, t_change, A))
        sigma_obs = sample("sigma_obs", dist.HalfNormal(scale=0.5))
        betas = sample("beta", dist.Normal(0, sigmas))
        mu = deterministic("mu", compute_mu(trend, X, betas, s_m, s_a))
        sample("obs", dist.Normal(mu, sigma_obs), obs=y)

    return model


@dataclass
class NumpyroModelData:
    # Time points, scaled between 0 and 1 (1 = last timestamp in history).
    t: jnp.ndarray
    # Response variable, shape like t
    y: jnp.ndarray
    # Capacity at each time point, shape like t. For logistic growth
    cap: jnp.ndarray
    # Time points at which changepoints occur
    t_change: jnp.ndarray
    # Indicator matrix for changepoints. shape (t.shape[0], t_change.shape[0])
    A: jnp.ndarray
    # Scale on the prior distribution for the changepoint deltas (same for all deltas)
    tau: float
    # Matrix of regressor values (seasonality, holidays, exogenous) at each time point, shape (t.shape[0], num_regressors)
    X: jnp.ndarray
    # Scale on the prior distribution for each regressor coefficient, shape (X.shape[1],)
    sigmas: jnp.ndarray
    # Indicator vector for regressors with additive effect, shape (X.shape[1],)
    s_a: jnp.ndarray
    # Indicator vector for regressors with multiplicative effect, shape (X.shape[1],)
    s_m: jnp.ndarray


@dataclass
class NumpyroModelParams:
    # Base rate of change of the trend function
    k: float
    # Offset for the trend function
    m: float
    # The difference applied to the rate of change at each changepoint. Size (num_changepoints, )
    delta: jnp.ndarray
    # The standard deviation of the response (y) at each timepoint. Same for all time points.
    sigma_obs: float
    # The coefficients of the seasonality, holidays, and exogenous regressors. Size (num_regressors,)
    beta: jnp.ndarray
