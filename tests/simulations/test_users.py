"""simulations/users.py test file."""
import math

from simulations.users import RandomUser


def test_random_user_no_trades() -> None:
    user = RandomUser(trade_probability=0., put_strikes=[.8, .9, 1., 1.1], call_strikes=[.9, 1., 1.1, 1.2])

    for _ in range(1000):
        trade = user.trade(1.)
        assert trade is None


def test_random_user_everytime_trade() -> None:
    user = RandomUser(trade_probability=1., put_strikes=[.8, .9, 1., 1.1], call_strikes=[.9, 1., 1.1, 1.2])

    for _ in range(1000):
        trade = user.trade(1.)
        assert trade is not None


def test_random_user_sometimes_trade() -> None:
    put_strikes = [.8, .9, 1., 1.1]
    call_strikes = [.9, 1., 1.1, 1.2]
    user = RandomUser(trade_probability=.5, put_strikes=put_strikes, call_strikes=call_strikes)

    trades = []
    for _ in range(1000):
        trade = user.trade(1.)
        trades.append((trade is None) * 1)
        if trade is not None:
            assert trade['type_'] in {'call', 'put'}
            assert trade['long_short'] in {'long', 'short'}
            if trade['type_'] == 'call':
                assert trade['strike_price'] in call_strikes
            else:
                assert trade['strike_price'] in put_strikes
            assert math.isclose(trade['quantity'], 1.)

    # There is incredibly small probability that this all user.trade() will be None or not None
    assert 0 < sum(trades) < 1000
