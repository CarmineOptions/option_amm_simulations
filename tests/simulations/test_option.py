"""simulations/option.py test file."""
import pytest

from simulations.option import Option


@pytest.mark.parametrize(
    'type_,long_short,fails',
    [
        ('call', 'long', False),
        ('call', 'short', False),
        ('put', 'long', False),
        ('put', 'short', False),
        ('long', 'short', True),
        ('put', 'put', True),
    ]
)
def test_option_init(type_: str, long_short: str, fails: bool) -> None:
    if not fails:
        # option should be correctly initialized
        option = Option(
            strike_price=1.,
            type_=type_,
            long_short=long_short,
            locked_capital=1.1,
            quantity=1.1
        )
        assert isinstance(option, Option)
    else:
        with pytest.raises(ValueError):
            Option(
                strike_price=1.,
                type_=type_,
                long_short=long_short,
                locked_capital=1.1,
                quantity=1.1
            )
