"""simulations/price_time_series.py test file."""
import math

import numpy as np
import pandas as pd

from simulations.price_time_series import generate_price_volatility_process


def test_generate_price_variance_process() -> None:
    """The generation of price and volatility series is random."""
    price, volatility = generate_price_volatility_process()
    assert isinstance(price, np.ndarray)
    assert isinstance(volatility, np.ndarray)
    assert price.shape == (10_000,)
    assert volatility.shape == (10_000,)
    price, volatility = pd.Series(price), pd.Series(volatility)

    assert not price.isna().any()
    assert not volatility.isna().any()

    assert (price > 0).all()
    assert (volatility > 0).all()
