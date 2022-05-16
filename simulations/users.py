from typing import Any, Dict, List, Optional
import math

import numpy as np
import random

from simulations.amm import AMM


class RandomUser:

    def __init__(self, trade_probability: float, put_strikes: List[float], call_strikes: List[float]) -> None:
        self.trade_probability = trade_probability
        self.put_strikes = put_strikes
        self.call_strikes = call_strikes

    def trade(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
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


class TraderUser:

    # - first sees the volatility that generated the process and believes that it should be 10% lower
    # - second sees the volatility that generated the process and believes it is the true one
    # - third sees the volatility that generated the process and believes that it should be 10% higher

    def __init__(
            self,
            amm: AMM,
            volatility_adjustment: float
    ) -> None:
        if volatility_adjustment < -1:
            raise ValueError

        self.amm = amm
        self.volatility_adjustment = volatility_adjustment

    def trade(self, current_price: float, current_volatility: float) -> Optional[Dict[str, Any]]:
        """
        Trader randomly orders all options and goes one by one.

        If trader sees profitable option (based on current adjusted volatility) it executes given option with
        probability (amm.premia - trader.premia) / trader.premia for shorts and with opposite sign for longs.
        """
        if not math.isclose(self.amm.current_underlying_price, current_price, rel_tol=0.00001):
            raise ValueError(
                f'amm current price: {self.amm.current_underlying_price}'
                f' should be equal to current_price {current_price}'
            )

        adjusted_volatility = current_volatility * (1 + self.volatility_adjustment)
        # fake_amm is used to calculate the "what if premia"...
        fake_amm = AMM(
            time_till_maturity=self.amm.time_till_maturity,
            current_underlying_price=self.amm.current_underlying_price,
            call_strikes=self.amm.call_strikes,
            put_strikes=self.amm.put_strikes,
            call_volatility=adjusted_volatility,
            put_volatility=adjusted_volatility,
            call_pool_size=self.amm.call_pool_size,
            put_pool_size=self.amm.put_pool_size,
        )

        types = ['call', 'put']
        long_shorts = ['long', 'short']
        np.random.shuffle(types)
        np.random.shuffle(long_shorts)

        for type_ in types:
            for long_short in long_shorts:
                strike_prices = self.amm.call_strikes if type_ == 'call' else self.amm.put_strikes
                np.random.shuffle(strike_prices)

                for strike_price in strike_prices:
                    amm_premia = self.amm.get_premia(strike_price, type_, long_short)
                    trader_premia = fake_amm.get_premia(strike_price, type_, long_short)
                    profitability = (amm_premia - trader_premia) / trader_premia

                    if 0 < profitability and long_short == 'short':
                        # if profitability is > 1... user believes the option has double the price
                        if random.random() < profitability:
                            return {
                                'type_': type_,
                                'long_short': long_short,
                                'strike_price': strike_price,
                                'quantity': 1.
                            }
                    elif profitability < 0 and long_short == 'long':
                        # if profitability is < -1... user believes the option should have double the price
                        if random.random() < -profitability:
                            return {
                                'type_': type_,
                                'long_short': long_short,
                                'strike_price': strike_price,
                                'quantity': 1.
                            }
        return None
