from typing import Any, Dict, List, Optional

import random


class RandomUser:

    def __init__(self, trade_probability: float, put_strikes: List[float], call_strikes: List[float]) -> None:
        self.trade_probability = trade_probability
        self.put_strikes = put_strikes
        self.call_strikes = call_strikes

    def trade(self, current_price: float) -> Optional[Dict[str, Any]]:
        # FIXME: drop the current_price, it is redundant
        if random.random() < self.trade_probability:
            # user trades
            type_ = random.choice(['call', 'put'])
            long_short = random.choice(['long', 'short'])
            if type_ == 'call':
                strike_price = random.choice(self.call_strikes)
            else:
                strike_price = random.choice(self.put_strikes)
            return {
                'type_': type_,
                'long_short': long_short,
                'strike_price': strike_price,
                'quantity': 1.
            }
        return None
