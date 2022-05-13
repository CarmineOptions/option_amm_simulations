"""simulations/option.py test file."""
import math
import pytest

from simulations.option import Option
from simulations.amm import AMM, black_scholes, NotEnoughPoolCapitalError


@pytest.mark.parametrize(
    'vol,s,k,r,t,expected_call_premia,expected_put_premia',
    [
        (20/100, 100, 100, 20/100, 0.128767, 4.277304453686696, 1.7348434806567639),
        (20/100, 100, 100, 20/100, 0.257534, 6.95278811264042, 1.932507244574353),
    ]
)
def test_black_scholes(
        vol: float,
        s: float,
        k: float,
        r: float,
        t: int,
        expected_call_premia: float,
        expected_put_premia: float
) -> None:
    call_premia, put_premia = black_scholes(vol=vol, s=s, k=k, r=r, t=t)

    assert math.isclose(call_premia, expected_call_premia, rel_tol=0.001)
    assert math.isclose(put_premia, expected_put_premia, rel_tol=0.001)

def test_amm_init() -> None:
    amm = AMM(time_till_maturity=10., current_underlying_price=1.)

    assert all(list(
        math.isclose(strike, call_strike, abs_tol=0.00001)
        for strike, call_strike in zip([x / 10 for x in range(9, 20)], amm.call_strikes)
    ))
    assert all(list(
        math.isclose(strike, put_strike, abs_tol=0.00001)
        for strike, put_strike in zip([x / 10 for x in range(2, 12)], amm.put_strikes)
    ))

    assert math.isclose(amm.call_volatility, 0.1, abs_tol=0.0000001)
    assert math.isclose(amm.put_volatility, 0.1, abs_tol=0.0000001)

    assert math.isclose(amm.call_pool_size, 100., abs_tol=0.0000001)
    assert math.isclose(amm.put_pool_size, 100., abs_tol=0.0000001)

    assert not amm.call_issued_options
    assert not amm.put_issued_options

    assert math.isclose(amm.time_till_maturity, 10., abs_tol=0.0000001)
    assert math.isclose(amm.current_underlying_price, 1., abs_tol=0.0000001)




# @pytest.mark.parametrize(
#     'vol,s,k,r,t,expected_call_premia,expected_put_premia',
#     [
#         (20/100, 100, 100, 20/100, 0.128767, 4.277304453686696, 1.7348434806567639),
#         (20/100, 100, 100, 20/100, 0.257534, 6.95278811264042, 1.932507244574353),
#     ]
# )
# def test_black_scholes(
