"""simulations/option.py test file."""
import math
import pytest
from unittest.mock import MagicMock

from simulations.option import Option
from simulations.amm import AMM, black_scholes, NotEnoughPoolCapitalError


REVERSE_SIDE = {'long': 'short', 'short': 'long'}


@pytest.mark.parametrize(
    'vol,s,k,r,t,expected_call_premia,expected_put_premia',
    [
        # risk free rate and time till maturity annualized
        (.2, 100, 100, .2, 0.128767, 4.277304453686696, 1.7348434806567639),
        (.2, 100, 100, .2, 0.257534, 6.95278811264042, 1.932507244574353),
        # time till maturity in hours... 100 hours (periods) is roughly equal to 0.011 year
        (.02, 100, 100, 0., 100, 7.965567455405804, 7.965567455405804)
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


def test_amm_next_epoch() -> None:
    amm = AMM(time_till_maturity=100., current_underlying_price=1.)
    assert math.isclose(amm.time_till_maturity, 100., abs_tol=0.0000001)
    assert math.isclose(amm.current_underlying_price, 1., abs_tol=0.0000001)

    amm.next_epoch(time_till_maturity=98., current_underlying_price=2.)
    assert math.isclose(amm.time_till_maturity, 98., abs_tol=0.0000001)
    assert math.isclose(amm.current_underlying_price, 2., abs_tol=0.0000001)

    with pytest.raises(ValueError):
        amm.next_epoch(time_till_maturity=-2., current_underlying_price=2.)

    with pytest.raises(ValueError):
        amm.next_epoch(time_till_maturity=1., current_underlying_price=0.)

    with pytest.raises(ValueError):
        amm.next_epoch(time_till_maturity=1., current_underlying_price=-2.)


@pytest.mark.parametrize(
    'strike_price, type_, long_short, quantity, time_till_maturity, current_underlying_price,'
    'call_volatility, put_volatility, call_pool_size, put_pool_size, expected_premia',
    [
        # The difference between those two put/call options in resulting premia is because for call
        # it is measure in base token (in case of eth/usdc, the eth is base) and for put it is measured
        # in quote token.
        (100., 'call', 'long', 1., 100., 100., 0.01, 0.01, 100, 10_000, 0.04128332503202088),
        (100., 'put', 'long', 1., 100., 100., 0.01, 0.01, 100, 10_000, 4.128332503202088),
        # Difference between shorts and longs is because of collected fees.
        (100., 'call', 'short', 1., 100., 100., 0.01, 0.01, 100, 10_000, 0.03848803682233148),
        (100., 'put', 'short', 1., 100., 100., 0.01, 0.01, 100, 10_000, 3.8488036822331475),
        # Double the quantity results in little over double the premia for long and little below for short
        (100., 'call', 'long', 2., 100., 100., 0.01, 0.01, 100, 10_000, 0.08300286311512753),
        (100., 'put', 'long', 2., 100., 100., 0.01, 0.01, 100, 10_000, 8.300286311512753),
        (100., 'call', 'short', 2., 100., 100., 0.01, 0.01, 100, 10_000, 0.07658957582940412),
        (100., 'put', 'short', 2., 100., 100., 0.01, 0.01, 100, 10_000, 7.6589575829404115),
        # lower time till maturity decreases the option premia, just a little more than by sqrt(old_ttm/new_ttm)
        (100., 'call', 'long', 1., 50., 100., 0.01, 0.01, 100, 10_000, 0.029197861998470978,),
        (100., 'put', 'long', 1., 50., 100., 0.01, 0.01, 100, 10_000, 2.9197861998470978,),
        (100., 'call', 'short', 1., 50., 100., 0.01, 0.01, 100, 10_000, 0.0272207642878838,),
        (100., 'put', 'short', 1., 50., 100., 0.01, 0.01, 100, 10_000, 2.7220764287883803),
        # increasing price of underlying asset increases price of call options and decreases price of put options
        (100., 'call', 'long', 1., 100., 110., 0.01, 0.01, 100, 10_000, 0.10269583569324303,),
        (100., 'put', 'long', 1., 100., 110., 0.01, 0.01, 100, 10_000, 0.997974581235503,),
        (100., 'call', 'short', 1., 100., 110., 0.01, 0.01, 100, 10_000, 0.09647717382987635,),
        (100., 'put', 'short', 1., 100., 110., 0.01, 0.01, 100, 10_000, 0.9112083661223075,),
        # doubling call volatility increases price of call options to almost double, keeps puts intact
        (100., 'call', 'long', 1., 100., 100., 0.02, 0.01, 100, 10_000, 0.08246253935824129),
        (100., 'put', 'long', 1., 100., 100., 0.02, 0.01, 100, 10_000, 4.128332503202088),
        (100., 'call', 'short', 1., 100., 100., 0.02, 0.01, 100, 10_000, 0.07688095073616794),
        (100., 'put', 'short', 1., 100., 100., 0.02, 0.01, 100, 10_000, 3.8488036822331475),
        # doubling put volatility increases price of put options to almost double, keeps calls intact
        (100., 'call', 'long', 1., 100., 100., 0.01, 0.02, 100, 10_000, 0.04128332503202088),
        (100., 'put', 'long', 1., 100., 100., 0.01, 0.02, 100, 10_000, 8.246253935824129),
        (100., 'call', 'short', 1., 100., 100., 0.01, 0.02, 100, 10_000, 0.03848803682233148),
        (100., 'put', 'short', 1., 100., 100., 0.01, 0.02, 100, 10_000, 7.688095073616793),
        # doubling call_pool_size decreases the impact of trade on volatility hence decreases price for long,
        # increases for short -> it is better for the trader since trader pays less or receives more
        # puts are same
        (100., 'call', 'long', 1., 100., 100., 0.01, 0.01, 200, 10_000, 0.04117757536416448),
        (100., 'put', 'long', 1., 100., 100., 0.01, 0.01, 200, 10_000, 4.128332503202088),
        (100., 'call', 'short', 1., 100., 100., 0.01, 0.01, 200, 10_000, 0.038584660375579775),
        (100., 'put', 'short', 1., 100., 100., 0.01, 0.01, 200, 10_000, 3.8488036822331475),
        # doubling put_pool_size decreases the impact of trade on volatility hence decreases price for long,
        # increases for short -> it is better for the trader since trader pays less or receives more
        # calls are same
        (100., 'call', 'long', 1., 100., 100., 0.01, 0.01, 100, 20_000, 0.04128332503202088),
        (100., 'put', 'long', 1., 100., 100., 0.01, 0.01, 100, 20_000, 4.117757536416447),
        (100., 'call', 'short', 1., 100., 100., 0.01, 0.01, 100, 20_000, 0.03848803682233148),
        (100., 'put', 'short', 1., 100., 100., 0.01, 0.01, 100, 20_000, 3.8584660375579776),
    ]
)
def test_get_premia(
        strike_price: float,
        type_: str,
        long_short: str,
        quantity: float,
        time_till_maturity: float,
        current_underlying_price: float,
        call_volatility: float,
        put_volatility: float,
        call_pool_size: float,
        put_pool_size: float,
        expected_premia: float
) -> None:
    amm = AMM(
        time_till_maturity=time_till_maturity,
        current_underlying_price=current_underlying_price,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        # if time_till_maturity is measured in hours,
        # multiply the volatility by sqrt(24*365) to get annualized volatility
        call_volatility=call_volatility,
        put_volatility=put_volatility,
        call_pool_size=call_pool_size,
        put_pool_size=put_pool_size,
    )

    assert math.isclose(
        amm.get_premia(strike_price, type_, long_short, quantity),
        expected_premia,
        rel_tol=0.0001
    )

@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_call_volatility, expected_call_pool_size',
    [
        (100., 'long', 1., 0.01010204081632653, 99.04128332503203),
        (100., 'short', 1., 0.009899999999999999, 99.96151196317767)
    ]
)
def test_trade_call_no_issued_options(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_call_volatility: float,
        expected_call_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )

    amm.trade(strike_price=strike_price, type_='call', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, expected_call_volatility, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, 10_000., rel_tol=0.00001)

    assert len(amm.call_issued_options) == 1
    assert math.isclose(amm.call_issued_options[0].strike_price, strike_price, rel_tol=0.00001)
    assert amm.call_issued_options[0].type_ == 'call'
    assert amm.call_issued_options[0].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.call_issued_options[0].locked_capital, quantity, rel_tol=0.00001)
    else:
        assert math.isclose(amm.call_issued_options[0].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.call_issued_options[0].quantity, quantity, rel_tol=0.00001)

    assert not amm.put_issued_options

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)

    # Trade the same option again
    amm.trade(strike_price=strike_price, type_='call', long_short=long_short, quantity=quantity)

    assert not math.isclose(amm.call_volatility, expected_call_volatility, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.00001)

    assert not math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, 10_000., rel_tol=0.00001)

    assert len(amm.call_issued_options) == 2
    assert math.isclose(amm.call_issued_options[1].strike_price, strike_price, rel_tol=0.00001)
    assert amm.call_issued_options[1].type_ == 'call'
    assert amm.call_issued_options[1].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.call_issued_options[1].locked_capital, quantity, rel_tol=0.00001)
    else:
        assert math.isclose(amm.call_issued_options[1].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.call_issued_options[1].quantity, quantity, rel_tol=0.00001)

    assert not amm.put_issued_options

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)


@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_put_volatility, expected_put_pool_size',
    [
        (90., 'long', 1., 0.01010204081632653, 9910.745187407312),
        (90., 'short', 1., 0.009899999999999999, 9999.319485242433)
    ]
)
def test_trade_put_no_issued_options(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_put_volatility: float,
        expected_put_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )

    amm.trade(strike_price=strike_price, type_='put', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, expected_put_volatility, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, 100., rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)

    assert len(amm.put_issued_options) == 1
    assert math.isclose(amm.put_issued_options[0].strike_price, strike_price, rel_tol=0.00001)
    assert amm.put_issued_options[0].type_ == 'put'
    assert amm.put_issued_options[0].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.put_issued_options[0].locked_capital, quantity * strike_price, rel_tol=0.00001)
    else:
        assert math.isclose(amm.put_issued_options[0].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.put_issued_options[0].quantity, quantity, rel_tol=0.00001)

    assert not amm.call_issued_options

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)

    # Trade the same option again
    amm.trade(strike_price=strike_price, type_='put', long_short=long_short, quantity=quantity)

    assert not math.isclose(amm.put_volatility, expected_put_volatility, rel_tol=0.00001)
    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.00001)

    assert not math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.call_pool_size, 100., rel_tol=0.00001)

    assert len(amm.put_issued_options) == 2
    assert math.isclose(amm.put_issued_options[1].strike_price, strike_price, rel_tol=0.00001)
    assert amm.put_issued_options[1].type_ == 'put'
    assert amm.put_issued_options[1].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.put_issued_options[1].locked_capital, quantity * strike_price, rel_tol=0.00001)
    else:
        assert math.isclose(amm.put_issued_options[1].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.put_issued_options[1].quantity, quantity, rel_tol=0.00001)

    assert not amm.call_issued_options

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)



@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_call_volatility, expected_call_pool_size',
    [
        (100., 'long', 1., 0.01010204081632653, 99.04128332503203),
        (100., 'short', 1., 0.009899999999999999, 99.96151196317767)
    ]
)
def test_trade_call_mismatching_issued_options(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_call_volatility: float,
        expected_call_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=110., type_='call', long_short='long', locked_capital=0., quantity=quantity)
    call_2 = Option(strike_price=110., type_='call', long_short='short', locked_capital=quantity, quantity=quantity)
    put_1 = Option(strike_price=80., type_='put', long_short='long', locked_capital=0., quantity=quantity)
    put_2 = Option(strike_price=80., type_='put', long_short='short', locked_capital=quantity * 90., quantity=quantity)
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_='call', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, expected_call_volatility, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, 10_000., rel_tol=0.00001)

    assert len(amm.call_issued_options) == 3
    assert amm.call_issued_options[0] == call_1
    assert amm.call_issued_options[1] == call_2
    assert math.isclose(amm.call_issued_options[2].strike_price, strike_price, rel_tol=0.00001)
    assert amm.call_issued_options[2].type_ == 'call'
    assert amm.call_issued_options[2].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        # means AMM is short
        assert math.isclose(amm.call_issued_options[2].locked_capital, quantity, rel_tol=0.00001)
    else:
        assert math.isclose(amm.call_issued_options[2].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.call_issued_options[2].quantity, quantity, rel_tol=0.00001)

    assert len(amm.put_issued_options) == 2
    assert amm.put_issued_options[0] == put_1
    assert amm.put_issued_options[1] == put_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)

    # Trade the same option again
    amm.trade(strike_price=strike_price, type_='call', long_short=long_short, quantity=quantity)

    assert not math.isclose(amm.call_volatility, expected_call_volatility, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.00001)

    assert not math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, 10_000., rel_tol=0.00001)

    assert len(amm.call_issued_options) == 4
    assert amm.call_issued_options[0] == call_1
    assert amm.call_issued_options[1] == call_2
    assert math.isclose(amm.call_issued_options[3].strike_price, strike_price, rel_tol=0.00001)
    assert amm.call_issued_options[3].type_ == 'call'
    assert amm.call_issued_options[3].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.call_issued_options[3].locked_capital, quantity, rel_tol=0.00001)
    else:
        assert math.isclose(amm.call_issued_options[3].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.call_issued_options[3].quantity, quantity, rel_tol=0.00001)

    assert len(amm.put_issued_options) == 2
    assert amm.put_issued_options[0] == put_1
    assert amm.put_issued_options[1] == put_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)


@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_put_volatility, expected_put_pool_size',
    [
        (90., 'long', 1., 0.01010204081632653, 9910.745187407312),
        (90., 'short', 1., 0.009899999999999999, 9999.319485242433)
    ]
)
def test_trade_put_mismatching_issued_options(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_put_volatility: float,
        expected_put_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=110., type_='call', long_short='long', locked_capital=0., quantity=quantity)
    call_2 = Option(strike_price=110., type_='call', long_short='short', locked_capital=quantity, quantity=quantity)
    put_1 = Option(strike_price=80., type_='put', long_short='long', locked_capital=0., quantity=quantity)
    put_2 = Option(strike_price=80., type_='put', long_short='short', locked_capital=quantity * 90., quantity=quantity)
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_='put', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, expected_put_volatility, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, 100., rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)

    assert len(amm.put_issued_options) == 3
    assert amm.put_issued_options[0] == put_1
    assert amm.put_issued_options[1] == put_2
    assert math.isclose(amm.put_issued_options[2].strike_price, strike_price, rel_tol=0.00001)
    assert amm.put_issued_options[2].type_ == 'put'
    assert amm.put_issued_options[2].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.put_issued_options[2].locked_capital, quantity * strike_price, rel_tol=0.00001)
    else:
        assert math.isclose(amm.put_issued_options[2].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.put_issued_options[2].quantity, quantity, rel_tol=0.00001)

    assert len(amm.call_issued_options) == 2
    assert amm.call_issued_options[0] == call_1
    assert amm.call_issued_options[1] == call_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)

    # Trade the same option again
    amm.trade(strike_price=strike_price, type_='put', long_short=long_short, quantity=quantity)

    assert not math.isclose(amm.put_volatility, expected_put_volatility, rel_tol=0.00001)
    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.00001)

    assert not math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.call_pool_size, 100., rel_tol=0.00001)

    assert len(amm.put_issued_options) == 4
    assert amm.put_issued_options[0] == put_1
    assert amm.put_issued_options[1] == put_2
    assert math.isclose(amm.put_issued_options[3].strike_price, strike_price, rel_tol=0.00001)
    assert amm.put_issued_options[3].type_ == 'put'
    assert amm.put_issued_options[3].long_short == REVERSE_SIDE[long_short]
    if long_short == 'long':
        assert math.isclose(amm.put_issued_options[3].locked_capital, quantity * strike_price, rel_tol=0.00001)
    else:
        assert math.isclose(amm.put_issued_options[3].locked_capital, 0., rel_tol=0.00001)
    assert math.isclose(amm.put_issued_options[3].quantity, quantity, rel_tol=0.00001)

    assert len(amm.call_issued_options) == 2
    assert amm.call_issued_options[0] == call_1
    assert amm.call_issued_options[1] == call_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)


@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_call_volatility, expected_call_pool_size',
    [
        (100., 'long', 1., 0.01010204081632653, 100.04128332503203),
        (100., 'short', 1., 0.009899999999999999, 100.96151196317767)
    ]
)
def test_trade_call_matching_issued_options_1(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_call_volatility: float,
        expected_call_pool_size: float
) -> None:
    """Entire option gets matched"""
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=100., type_='call', long_short='long', locked_capital=0., quantity=quantity)
    call_2 = Option(strike_price=100., type_='call', long_short='short', locked_capital=quantity, quantity=quantity)
    put_1 = Option(strike_price=100., type_='put', long_short='long', locked_capital=0., quantity=quantity)
    put_2 = Option(strike_price=100., type_='put', long_short='short', locked_capital=quantity * 100., quantity=quantity)
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_='call', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, expected_call_volatility, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.00001)

    # assert math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, 10_000., rel_tol=0.00001)

    assert len(amm.call_issued_options) == 1
    if long_short == 'long':
        assert amm.call_issued_options[0] == call_2
    else:
        assert amm.call_issued_options[0] == call_1

    assert len(amm.put_issued_options) == 2
    assert amm.put_issued_options[0] == put_1
    assert amm.put_issued_options[1] == put_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)



@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_put_volatility, expected_put_pool_size',
    [
        (90., 'long', 1., 0.01010204081632653, 10_000.745187407312),
        (90., 'short', 1., 0.009899999999999999, 10_089.319485242433)
    ]
)
def test_trade_put_matching_issued_options_1(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_put_volatility: float,
        expected_put_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=110., type_='call', long_short='long', locked_capital=0., quantity=quantity)
    call_2 = Option(strike_price=110., type_='call', long_short='short', locked_capital=quantity, quantity=quantity)
    put_1 = Option(strike_price=90., type_='put', long_short='long', locked_capital=0., quantity=quantity)
    put_2 = Option(strike_price=90., type_='put', long_short='short', locked_capital=quantity * 90., quantity=quantity)
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_='put', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, expected_put_volatility, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, 100., rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)

    assert len(amm.put_issued_options) == 1
    if long_short == 'long':
        assert amm.put_issued_options[0] == put_2
    else:
        assert amm.put_issued_options[0] == put_1

    assert len(amm.call_issued_options) == 2
    assert amm.call_issued_options[0] == call_1
    assert amm.call_issued_options[1] == call_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)


@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_call_volatility, expected_call_pool_size',
    [
        (100., 'long', 1., 0.01010204081632653, 100.04128332503203),
        (100., 'short', 1., 0.009899999999999999, 100.96151196317767)
    ]
)
def test_trade_call_matching_issued_options_2(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_call_volatility: float,
        expected_call_pool_size: float
) -> None:
    """Entire option gets matched"""
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=100., type_='call', long_short='long', locked_capital=0., quantity=3 * quantity)
    call_2 = Option(
        strike_price=100., type_='call', long_short='short', locked_capital=3 * quantity, quantity=3 * quantity
    )
    put_1 = Option(strike_price=100., type_='put', long_short='long', locked_capital=0., quantity=3 * quantity)
    put_2 = Option(
        strike_price=100., type_='put', long_short='short', locked_capital=3 * quantity * 100., quantity=3 * quantity
    )
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_='call', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, expected_call_volatility, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, 10_000., rel_tol=0.00001)

    assert len(amm.call_issued_options) == 2
    if long_short == 'long':
        assert amm.call_issued_options[0] == call_2
        assert amm.call_issued_options[1].strike_price == strike_price
        assert amm.call_issued_options[1].type_ == 'call'
        assert amm.call_issued_options[1].long_short == long_short
        assert math.isclose(amm.call_issued_options[1].locked_capital, 0., rel_tol=0.000001)
        assert math.isclose(amm.call_issued_options[1].quantity, 2 * quantity, rel_tol=0.000001)
    else:
        assert amm.call_issued_options[0] == call_1
        assert amm.call_issued_options[1].strike_price == strike_price
        assert amm.call_issued_options[1].type_ == 'call'
        assert amm.call_issued_options[1].long_short == long_short
        assert math.isclose(amm.call_issued_options[1].locked_capital, 2 * quantity, rel_tol=0.000001)
        assert math.isclose(amm.call_issued_options[1].quantity, 2 * quantity, rel_tol=0.000001)

    assert len(amm.put_issued_options) == 2
    assert amm.put_issued_options[0] == put_1
    assert amm.put_issued_options[1] == put_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)



@pytest.mark.parametrize(
    'strike_price, long_short, quantity, expected_put_volatility, expected_put_pool_size',
    [
        (90., 'long', 1., 0.01010204081632653, 10_000.745187407312),
        (90., 'short', 1., 0.009899999999999999, 10_089.319485242433)
    ]
)
def test_trade_put_matching_issued_options_2(
        strike_price: float,
        long_short: str,
        quantity: float,
        expected_put_volatility: float,
        expected_put_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=100., type_='call', long_short='long', locked_capital=0., quantity=3 * quantity)
    call_2 = Option(
        strike_price=100., type_='call', long_short='short', locked_capital=3 * quantity, quantity=3 * quantity
    )
    put_1 = Option(strike_price=90., type_='put', long_short='long', locked_capital=0., quantity=3 * quantity)
    put_2 = Option(
        strike_price=90., type_='put', long_short='short', locked_capital=3 * quantity * 90., quantity=3 * quantity
    )
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_='put', long_short=long_short, quantity=quantity)

    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.00001)
    assert math.isclose(amm.put_volatility, expected_put_volatility, rel_tol=0.00001)

    assert math.isclose(amm.call_pool_size, 100., rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)

    assert len(amm.put_issued_options) == 2
    if long_short == 'long':
        assert amm.put_issued_options[0] == put_2
        assert amm.put_issued_options[1].strike_price == strike_price
        assert amm.put_issued_options[1].type_ == 'put'
        assert amm.put_issued_options[1].long_short == long_short
        assert math.isclose(amm.put_issued_options[1].locked_capital, 0., rel_tol=0.000001)
        assert math.isclose(amm.put_issued_options[1].quantity, 2 * quantity, rel_tol=0.000001)
    else:
        assert amm.put_issued_options[0] == put_1
        assert amm.put_issued_options[1].strike_price == strike_price
        assert amm.put_issued_options[1].type_ == 'put'
        assert amm.put_issued_options[1].long_short == long_short
        assert math.isclose(amm.put_issued_options[1].locked_capital, 2 * quantity * strike_price, rel_tol=0.000001)
        assert math.isclose(amm.put_issued_options[1].quantity, 2 * quantity, rel_tol=0.000001)

    assert len(amm.call_issued_options) == 2
    assert amm.call_issued_options[0] == call_1
    assert amm.call_issued_options[1] == call_2

    assert math.isclose(amm.time_till_maturity, 100., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, 100., rel_tol=0.00001)


@pytest.mark.parametrize(
    'strike_price, type_, long_short, quantity, expected_call_volatility, expected_call_pool_size',
    [
        (100., 'call', 'long', 1., 0.01010204081632653, 99.04128332503203),
        (100., 'call', 'short', 1., 0.009899999999999999, 99.96151196317767),
        (90., 'put', 'long', 1., 0.01010204081632653, 9910.745187407312),
        (90., 'put', 'short', 1., 0.009899999999999999, 9999.319485242433)
    ]
)
def test_trade_call_matching_issued_options_not_enough_quantity(
        strike_price: float,
        type_: str,
        long_short: str,
        quantity: float,
        expected_call_volatility: float,
        expected_call_pool_size: float
) -> None:
    """Entire option gets matched"""
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=100., type_='call', long_short='long', locked_capital=0., quantity=quantity / 2)
    call_2 = Option(
        strike_price=100., type_='call', long_short='short', locked_capital=quantity / 2, quantity=quantity / 2
    )
    put_1 = Option(strike_price=100., type_='put', long_short='long', locked_capital=0., quantity=quantity / 2)
    put_2 = Option(
        strike_price=100., type_='put', long_short='short', locked_capital=quantity * 100 / 2., quantity=quantity / 2
    )
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    amm.trade(strike_price=strike_price, type_=type_, long_short=long_short, quantity=quantity)

    if type_ == 'call':
        assert len(amm.call_issued_options) == 3
        assert len(amm.put_issued_options) == 2
    else:
        assert len(amm.call_issued_options) == 2
        assert len(amm.put_issued_options) == 3


def test_clear_raises() -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=100., type_='call', long_short='long', locked_capital=0., quantity=100.)
    call_2 = Option(strike_price=100., type_='call', long_short='short', locked_capital=100., quantity=100.)
    put_1 = Option(strike_price=90., type_='put', long_short='long', locked_capital=0., quantity=100.)
    put_2 = Option(strike_price=90., type_='put', long_short='short', locked_capital=100. * 90., quantity=100.)
    amm.call_issued_options = [call_1, call_2]
    amm.put_issued_options = [put_1, put_2]

    with pytest.raises(ValueError):
        amm.clear()


@pytest.mark.parametrize(
    'current_underlying_price, expected_call_pool_size, expected_put_pool_size',
    [
        (
                80.,
                106.,
                10_660. - 4*40 + 3*30 - 10
        ),
        (
                95.,
                106.,
                10_660. - 4*25 + 3*15
        ),
        (
                105.,
                106. + 1*5/105 - 2*5/105,
                10_660. - 4*15 + 3*5
        ),
        (
                115.,
                106. + 1*15/115 - 2*15/115,
                10_660. - 4*5
        ),
        (
                125.,
                106. + 1*25/125 - 2*25/125 + 3*5/125,
                10_660.
        ),
        (
                135.,
                106. + 1*35/135 - 2*35/135 + 3*15/135 - 4*5/135,
                10_660.
        )
    ]
)
def test_clear(
        current_underlying_price: float,
        expected_call_pool_size: float,
        expected_put_pool_size: float
) -> None:
    amm = AMM(
        time_till_maturity=100.,
        current_underlying_price=100.,
        call_strikes=[float(x) for x in range(90, 160, 10)],
        put_strikes=[float(x) for x in range(50, 120, 10)],
        call_volatility=0.01,
        put_volatility=0.01,
        call_pool_size=100.,
        put_pool_size=10_000.,
    )
    call_1 = Option(strike_price=100., type_='call', long_short='long', locked_capital=0., quantity=1.)
    call_2 = Option(strike_price=100., type_='call', long_short='short', locked_capital=2., quantity=2.)
    call_3 = Option(strike_price=120., type_='call', long_short='long', locked_capital=0., quantity=3.)
    call_4 = Option(strike_price=130., type_='call', long_short='short', locked_capital=4., quantity=4.)
    put_1 = Option(strike_price=90., type_='put', long_short='long', locked_capital=0., quantity=1.)
    put_2 = Option(strike_price=90., type_='put', long_short='short', locked_capital=180., quantity=2.)
    put_3 = Option(strike_price=110., type_='put', long_short='long', locked_capital=0., quantity=3.)
    put_4 = Option(strike_price=120., type_='put', long_short='short', locked_capital=480., quantity=4.)
    amm.call_issued_options = [call_1, call_2, call_3, call_4]
    amm.put_issued_options = [put_1, put_2, put_3, put_4]

    amm.next_epoch(time_till_maturity=0., current_underlying_price=current_underlying_price)
    amm.clear()

    assert math.isclose(amm.call_volatility, 0.01, rel_tol=0.000001)
    assert math.isclose(amm.put_volatility, 0.01, rel_tol=0.000001)

    assert math.isclose(amm.call_pool_size, expected_call_pool_size, rel_tol=0.00001)
    assert math.isclose(amm.put_pool_size, expected_put_pool_size, rel_tol=0.00001)

    assert not amm.call_issued_options
    assert not amm.put_issued_options

    assert math.isclose(amm.time_till_maturity, 0., rel_tol=0.00001)
    assert math.isclose(amm.current_underlying_price, current_underlying_price, rel_tol=0.00001)

