from typing import Any, Dict


class Option:
    def __init__(
            self,
            strike_price: float,
            type_: str,
            long_short: str,
            locked_capital: float,
            quantity: float
    ) -> None:
        self.strike_price = strike_price
        if type_ not in {'call', 'put'}:
            raise ValueError
        self.type_ = type_
        if long_short not in {'long', 'short'}:
            raise ValueError
        self.long_short = long_short
        self.locked_capital = locked_capital
        self.quantity = quantity

    def __dict__(self) -> Dict[str, Any]:
        return {
            'strike_price': self.strike_price,
            'type_': self.type_,
            'long_short': self.long_short,
            'locked_capital': self.locked_capital,
            'quantity': self.quantity
        }

    def __str__(self) -> str:
        return f"""
            strike_price: {self.strike_price}
            type_: {self.type_}
            long_short: {self.long_short}
            locked_capital: {self.locked_capital}
            quantity: {self.quantity}
        """
