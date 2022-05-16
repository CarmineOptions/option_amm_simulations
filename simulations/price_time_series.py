from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats



def _calc_volatility(alpha: float, beta: float, sigma: np.array) -> np.array:
    """
    Volatility for Black Scholes model is "is the standard deviation of the stock's returns".

    std = sqrt(var)

    variance of the process (described in price_variance_process function) in time t
    based on https://stats.stackexchange.com/questions/256437/variance-of-a-stationary-ar2-model
    var_t = x_t/y
    where x_t = (1 - beta) * sigma_t
    and y = (1 + beta) * (1 - alpha - beta) * (1 + alpha - beta)
    """
    x = (1 - beta) * sigma
    y = (1 + beta) * (1 - alpha - beta) * (1 + alpha - beta)
    if 0.95 < alpha + beta:
        raise ValueError(
            f'alpha + beta = {alpha + beta}, which does not leave enough room to calculate true variance of the process'
        )
    return np.sqrt(x / y)


def generate_price_volatility_process(
        alpha: float = 0.3,
        beta: float = 0.1,
        gamma: float = 0.9,
        series_len: int = 10_000,
        epsilon_mean: float = 0.,
        error_var: float = 0.002,
        initial_sigma: float = 0.05,
        initial_price: float = 1.,
) -> Tuple[np.array, np.array]:
    """
    Returns the series of prices and series of true volatility.

    Price process is described as
    price_t = (1 + r_t) * price_{t-1}
    where
    r_t = alpha * r_{t-1} + beta * r_{t_2} + epsilon_t
    Where
        epsilon_t ~ N(epsilon_mean, sigma_t)
        sigmq_t ~ gamma * sigma_{t-1} + e_t
        e_t ~ U[0, error_var]
    """
    e = scipy.stats.uniform.rvs(0, error_var, series_len)
    latest_sigma = initial_sigma
    sigma = []
    for e_t in e:
        latest_sigma = gamma * latest_sigma + e_t
        sigma.append(latest_sigma)
    # by design, all sigma values are positive
    sigma = np.array(sigma)

    epsilon = []
    for sigma_t in sigma:
        epsilon_t = scipy.stats.norm.rvs(epsilon_mean, sigma_t, 1)[0]
        epsilon.append(epsilon_t)
    epsilon = np.array(epsilon)

    r_1 = 0
    r_2 = 0
    r = []
    for i, epsilon_t in enumerate(epsilon):
        r_t = alpha * r_1 + beta * r_2 + epsilon_t
        r.append(r_t)
        r_2 = r_1
        r_1 = r_t
    r = np.array(r)

    price_t = initial_price
    price = []
    for r_t in r:
        price_t = (1 + r_t) * price_t
        price.append(price_t)
    price = np.array(price)

    return price, _calc_volatility(alpha, beta, sigma)
