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
