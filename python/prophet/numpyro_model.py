from abc import ABC, abstractmethod
from dataclasses import dataclass

from jax import jit
from jax.lax import scan
import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
import numpyro.distributions as dist
from numpyro import sample, plate, deterministic

from typing import Callable, Literal, Optional, Tuple


class Trend(ABC):
    def __init__(
        self,
        k: float,
        m: float,
        delta: jnp.ndarray,
        t: jnp.ndarray,
        cap: Optional[jnp.ndarray],
        A: jnp.ndarray,
        t_change: jnp.ndarray,
        S: int,
    ):
        """
        Parameters
        ----------
        k: The base growth rate.
        m: The offset parameter for the trend curve.
        delta: An array of length equal to the number of changepoints. Each element is the change in the growth rate at t_change.
        t: An array of normalized time values.
        cap: The capacity at each time point t.
        A: The changepoint matrix.
        t_change: The normalized time values at which changepoints are assumed to occur.
        S: The number of changepoints.
        """
        self.k = k
        self.m = m
        self.delta = delta
        self.t = t
        self.cap = cap
        self.A = A
        self.t_change = t_change
        self.S = S

    @abstractmethod
    def compute_values(self) -> jnp.ndarray:
        pass


@jit
def linear_trend(k, m, delta, t, t_change, A) -> jnp.ndarray:
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


def compute_gamma(k, m, delta, t_change, S) -> jnp.ndarray:
    """
    Calculates the offset parameter adjustments (gamma_j) required at each changepoint assuming logistic growth.
    The equation for gamma_j is defined in page 9 of the original Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf.

    Parameters
    ----------
    k: The base growth rate.
    m: The offset parameter for the trend curve.
    delta: An array of length equal to the number of changepoints. Each element is the change in the growth rate at t_change.
    t_change: The normalized time values at which changepoints are assumed to occur.
    S: The number of changepoints.

    Returns
    -------
    An array of offset parameter adjustments required at each changepoint.
    """
    k_s = jnp.append(jnp.array(k), k + jnp.cumsum(delta))
    _, gamma = scan(transition_function, (m, t_change, k_s), jnp.arange(S))
    return gamma


@jit
def logistic_trend(k, m, delta, t, cap, t_change, S, A) -> jnp.ndarray:
    """
    Calculates the trend value at each time point t assuming logistic growth.
    The trend function g(t) is defined in equation 3 on page 9 of the original Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf.

    Returns
    -------
    An array of trend values at each time point.
    """
    gamma = compute_gamma(k, m, delta, t_change, S)
    inv_logit = expit(jnp.multiply(k + A.dot(delta), (t - (m + A.dot(gamma)))))
    return jnp.multiply(cap, inv_logit)


@jit
def flat_trend(m, T) -> jnp.ndarray:
    return jnp.repeat(m, T)


def construct_changepoint_matrix(
    t: np.ndarray, t_change: np.ndarray, T: int, S: int
) -> jnp.ndarray:
    """
    Constructs the A(t) indicator matrix for changepoints as defined in page 9 of the original
    Forecasting at Scale paper: https://peerj.com/preprints/3190.pdf

    Parameters
    ----------
    t: An array of normalized time values.
    t_change: The normalized time values at which changepoints are assumed to occur.
    T: The number of time points.
    S: The number of changepoints.

    Returns
    -------
    A matrix of shape (number of time points, number of changepoints). For each row (i.e. each time point),
    the value of column j is 1 if changepoint j has taken effect by that time point, and 0 otherwise.
    """
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
def compute_mu(
    trend: jnp.ndarray, X: jnp.ndarray, betas: jnp.ndarray, s_m: jnp.ndarray, s_a: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculates the expected value of the response variable y at each time point t.

    Parameters
    ----------
    trend: The trend value at each time point t.
    X: The matrix containing fourier values representing seasonality, holidays indicators, and exogenous regressor values.
    betas: The coefficients for the features in X.
    s_m: An indicator matrix for the features in X that have a multiplicative effect on y.
    s_a: An indicator matrix for the features in X that have an additive effect on y.
    """
    return jnp.multiply(trend, (1 + X.dot(betas * s_m))) + X.dot(betas * s_a)


def model(T, K, t, cap, y, S, t_change, X, sigmas, tau, trend_indicator, s_a, s_m) -> None:
    A = construct_changepoint_matrix(np.array(t), np.array(t_change), T, S)
    k = sample("k", dist.Normal(0, 5))
    m = sample("m", dist.Normal(0, 5))
    with plate("delta_dim", S):
        delta = sample("delta", dist.Laplace(0, tau))
    if trend_indicator == 0:
        trend = deterministic("trend", linear_trend(k, m, delta, t, t_change, A))
    elif trend_indicator == 1:
        trend = deterministic("trend", logistic_trend(k, m, delta, t, cap, t_change, S, A))
    elif trend_indicator == 2:
        trend = deterministic("trend", flat_trend(m, T))
    else:
        raise ValueError("Invalid trend_indicator value")
    sigma_obs = sample("sigma_obs", dist.HalfNormal(scale=0.5))
    betas = sample("beta", dist.MultivariateNormal(0, jnp.diag(sigmas)))
    mu = deterministic("mu", compute_mu(trend, X, betas, s_m, s_a))
    with plate("data", T):
        sample("obs", dist.Normal(mu, sigma_obs), obs=y)


@dataclass
class NumpyroModelInput:
    # Number of time points
    T: int
    # Number of regressors (seasonality components, holidays, extra regressors)
    K: int
    # Time points, shape (T,)
    t: jnp.ndarray
    # Capacity at each time point, shape (T,). For logistic growth
    cap: jnp.ndarray
    # Response variable, shape (T,)
    y: jnp.ndarray
    # Number of changepoints
    S: int
    # Time points at which changepoints occur, shape (S,)
    t_change: jnp.ndarray
    # Matrix of regressor values at each time point, shape (T, K)
    X: jnp.ndarray
    # Scale on the prior distribution for each regressor coefficient, shape (K,)
    sigmas: jnp.ndarray
    # Scale on the prior distribution for the changepoint deltas (same for all deltas)
    tau: float
    # Indicator for the trend function to use (0: linear, 1: logistic, 2: flat)
    trend_indicator: Literal[0, 1, 2]
    # Indicator matrix for regressors with additive effect, shape (K,)
    s_a: jnp.ndarray
    # Indicator matrix for regressors with multiplicative effect, shape (K,)
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
