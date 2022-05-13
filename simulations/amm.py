from typing import Any, Dict, List, Tuple
import math

import scipy.stats

from simulations.option import Option


def black_scholes(vol: float, s: float, k: float, r: float, t: int) -> Tuple[float, float]:
    d_1 = 1 / math.sqrt(t) / vol * (math.log(s / k) + (r + vol ** 2 / 2) * t)
    d_2 = d_1 - vol * math.sqrt(t)

    cdf = scipy.stats.norm.cdf

    call_premia = cdf(d_1) * s - cdf(d_2) * k * math.exp(-r * t)
    put_premia = k * math.exp(-r * t) - s + call_premia

    return call_premia, put_premia


class NotEnoughPoolCapitalError(Exception):
    pass


class AMM:
    # FEE_SIZE is relative fee from paid/received premia
    FEE_SIZE = 0.03

    # ALPHA determines the speed of volatility adjustments
    ALPHA = 1

    RISK_FREE_RATE = 0.

    REVERSE_LONG_SHORT = {'long': 'short', 'short': 'long'}

    def __init__(self, time_till_maturity: int, current_underlying_price: float) -> None:
        self.call_strikes = [x / 10 for x in range(9, 20)]
        self.put_strikes = [x / 10 for x in range(2, 12)]

        self.call_volatility = 0.1
        self.put_volatility = 0.1

        self.call_pool_size = 100
        self.put_pool_size = 100

        self.call_issued_options = []
        self.put_issued_options = []

        self.time_till_maturity = time_till_maturity
        self.current_underlying_price = current_underlying_price

    def next_epoch(self, time_till_maturity: int, current_underlying_price: float) -> None:
        if time_till_maturity < 1:
            raise ValueError
        self.time_till_maturity = time_till_maturity
        self.current_underlying_price = current_underlying_price

    def _get_new_volatility(self, type_: str, long_short: str, quantity: float) -> float:
        """
        See https://carmine-finance.gitbook.io/carmine-options-amm/mechanics-deeper-look/option-pricing-mechanics#volatility-updates
        """
        if type_ == 'put':
            correct_token_quantity = quantity * self.current_underlying_price
        else:
            correct_token_quantity = quantity
        if type_ == 'call':
            current_volatility = self.call_volatility
            new_pool_size = self.call_pool_size - correct_token_quantity
        else:
            current_volatility = self.put_volatility
            new_pool_size = self.put_pool_size -correct_token_quantity

        signed_quantity = correct_token_quantity if long_short=='long' else -correct_token_quantity

        new_volatility = current_volatility / (1 - (signed_quantity / new_pool_size) ** self.ALPHA)

        return new_volatility

    def _get_trade_volatility(self, type_: str, long_short: str, quantity: float) -> float:
        current_volatility = self.call_volatility if type_ == 'call' else self.put_volatility
        new_volatility = self._get_new_volatility(type_, long_short, quantity)
        return (current_volatility + new_volatility) / 2

    def _get_price(self, strike: float, type_: str, long_short: str, quantity: float) -> float:
        trade_volatility = self._get_trade_volatility(type_, long_short, quantity)

        vol = trade_volatility
        t = self.time_till_maturity
        r = self.RISK_FREE_RATE
        s = self.current_underlying_price
        k = strike

        call_premia, put_premia = black_scholes(vol, s, k, r, t)

        # call premia is paid in base token, hence:
        call_premia = call_premia / self.current_underlying_price

        if type_ == 'call':
            return call_premia
        return put_premia

    def _update_volatility(self, type_: str, long_short: str, quantity: float) -> None:
        if type_ == 'call':
            self.call_volatility = self._get_new_volatility(type_, long_short, quantity)
        else:
            self.put_volatility = self._get_new_volatility(type_, long_short, quantity)

    def get_premia(self, strike_price: float, type_: str, long_short: str, quantity: float = 1.) -> float:
        premia = self._get_price(strike=strike_price, type_=type_, long_short=long_short, quantity=quantity)
        if long_short == 'long':
            # User goes long -> pool is short and receives premia
            return premia * (1 + self.FEE_SIZE) * quantity
        # User goes short -> pool is long and pays premia
        return premia * (1 - self.FEE_SIZE) * quantity

    def _find_options(self, strike_price: float, type_: str, long_short: str) -> List[Option]:
        """Looks if given option is owned by the AMM, if none exists return None"""
        if type_ not in {'call', 'put'}:
            raise ValueError
        if long_short not in {'long', 'short'}:
            raise ValueError

        if type_ == 'call':
            options = [
                option
                for option in self.call_issued_options
                if option.long_short == long_short
                if math.isclose(option.strike_price, strike_price, abs_tol=0.001)
            ]
        else:
            options = [
                option
                for option in self.put_issued_options
                if option.long_short == long_short
                if math.isclose(option.strike_price, strike_price, abs_tol=0.001)
            ]
        return options

    def _remove_option(self, option: Option) -> None:
        if option.type_ == 'call':
            size_before = len(self.call_issued_options)
            self.call_issued_options.remove(option)
            size_after = len(self.call_issued_options)
            if size_before - size_after != 1:
                raise ValueError
            return
        size_before = len(self.put_issued_options)
        self.put_issued_options.remove(option)
        size_after = len(self.put_issued_options)
        if size_before - size_after != 1:
            raise ValueError

    def _add_option(self, option: Option) -> None:
        if option.type_ == 'call':
            self.call_issued_options.append(option)
        else:
            self.put_issued_options.append(option)

    def _pay_receive_premia(self, type_: str, signed_premia_after_fee: float) -> None:
        if type_ == 'call':
            self.call_pool_size += signed_premia_after_fee
        else:
            self.put_pool_size += signed_premia_after_fee

    def trade(self, strike_price: float, type_: str, long_short: str, quantity: float) -> Option:
        """
        strike_price, type_, long_short - correspond TO WHAT THE USER WANTS TO DO!!!

        First look if given option is owned by the AMM.
        If no option is found new one is created for user and returned and opposite one is added to the pool.
        """
        if type_ not in {'call', 'put'}:
            raise ValueError
        if long_short not in {'long', 'short'}:
            raise ValueError
        if type_ == 'call':
            if not [strike for strike in self.call_strikes if math.isclose(strike, strike_price, abs_tol=0.001)]:
                raise ValueError
        else:
            if not [strike for strike in self.put_strikes if math.isclose(strike, strike_price, abs_tol=0.001)]:
                raise ValueError

        existing_options = self._find_options(strike_price, type_, long_short)

        # 1) get_premia
        # TODO: FEES ARE VIRTUAL AND ARE NOT "REMOVED" FROM TRADERS
        premia_after_fee = self.get_premia(strike_price, type_, long_short)
        # If user goes long, pool receives fee, otherwise pays it
        signed_premia_after_fee = premia_after_fee if long_short == 'long' else -premia_after_fee

        # FIXME: account for options already in pool
        # 2.1) check enough capital in pool in case of short option (user short)
        if long_short == 'short':
            if type_ == 'call':
                if self.call_pool_size < premia_after_fee:
                    raise NotEnoughPoolCapitalError
            if type_ == 'put':
                if self.put_pool_size < premia_after_fee:
                    raise NotEnoughPoolCapitalError
        # 2.2) In case of long (user long) check enough capital
        if long_short == 'long':
            if type_ == 'call':
                if self.call_pool_size < quantity:
                    raise NotEnoughPoolCapitalError
            if type_ == 'put':
                if self.put_pool_size < quantity * self.current_underlying_price:
                    raise NotEnoughPoolCapitalError

        # 3) adjust volatility
        self._update_volatility(type_, long_short, quantity=quantity)

        # 4) pay/receive premia
        self._pay_receive_premia(type_, signed_premia_after_fee)

        all_locked_capital = sum(option.locked_capital for option in existing_options)
        all_locked_capital_base = all_locked_capital if type_=='call' else all_locked_capital/self.current_underlying_price
        # FIXME: locked capital is strike*quantity for puts
        if existing_options and (all_locked_capital_base > quantity):
            # 5.1) Aggregate existing_options into one:
            # put the options together, so that they cover the required option
            # quantity is in base (ETH) token, call's locked capital in base (ETH) and put's in (USDC)
            existing_option = Option(
                strike_price=existing_options[0].strike_price,
                type_=existing_options[0].type_,
                long_short=existing_options[0].long_short,
                locked_capital=quantity if type_=='call' else quantity*self.current_underlying_price,
                quantity=quantity
            )
            if type_ == 'call':
                remaining_locked_capital = all_locked_capital - quantity
                remaining_quantity = all_locked_capital - quantity
            else:
                remaining_locked_capital = all_locked_capital - quantity*self.current_underlying_price
                remaining_quantity = all_locked_capital / self.current_underlying_price - quantity
            complementary_option = Option(
                strike_price=existing_options[0].strike_price,
                type_=existing_options[0].type_,
                long_short=existing_options[0].long_short,
                locked_capital=remaining_locked_capital,
                quantity=remaining_quantity
            )

            # 5.2) remove existing_options and add existing_option and complementary_option
            for option in existing_options:
                self._remove_option(option)
            self._add_option(existing_option)
            self._add_option(complementary_option)

            # 5) remove option
            self._remove_option(existing_option)

            # 6) unlock capital (if some is locked)
            if long_short == 'short':
                # pool has to lock in capital
                if type_ == 'call':
                    self.call_pool_size += quantity
                else:
                    self.put_pool_size += quantity * self.current_underlying_price

            return existing_option

        else:  # redundant else, but makes it easier to read
            # 5) issue option
            if type_ == 'call':
                locked_capital = quantity
            else:
                locked_capital = quantity * self.current_underlying_price
            if long_short == 'long':
                user_option = Option(strike_price, type_, long_short, locked_capital=0., quantity=quantity)
                pool_option = Option(
                    strike_price,
                    type_,
                    self.REVERSE_LONG_SHORT[long_short],
                    locked_capital=locked_capital,
                    quantity=quantity
                )
            else:
                user_option = Option(strike_price, type_, long_short, locked_capital=locked_capital, quantity=quantity)
                pool_option = Option(
                    strike_price,
                    type_,
                    self.REVERSE_LONG_SHORT[long_short],
                    locked_capital=0,
                    quantity=quantity
                )

            # 6) add AMM's option to pool and return option
            self._add_option(pool_option)

            # 7) lock capital
            if long_short == 'long':
                # pool has to lock in capital since the pool is underwriting (user is long)
                if type_ == 'call':
                    self.call_pool_size -= quantity
                else:
                    self.put_pool_size -= quantity * self.current_underlying_price

            return user_option

    def clear(self) -> None:
        """Executes all options with current self.current_underlying_price."""
        for call_option in list(self.call_issued_options):
            # call's locked capital in base (ETH)
            # call pool is in base (ETH)
            if call_option.long_short == 'long':
                if self.current_underlying_price > call_option.strike_price:
                    # if in the money, pool gets profit
                    profit = (self.current_underlying_price - call_option.strike_price) / self.current_underlying_price
                    self.call_pool_size += call_option.quantity * profit
                else:
                    # if out of money, pool gets nothing
                    self.call_pool_size += 0
            else:
                if self.current_underlying_price > call_option.strike_price:
                    # if in the money, pool gets only part of the locked capital
                    loss = (self.current_underlying_price - call_option.strike_price) / self.current_underlying_price
                    self.call_pool_size += call_option.locked_capital - call_option.quantity * loss
                else:
                    # if out of money
                    self.call_pool_size += call_option.locked_capital
            self._remove_option(call_option)

        for put_option in list(self.put_issued_options):
            # put's locked capital in quote (USDC)
            # put pool is in quote (USDC)
            if put_option.long_short == 'long':
                if self.current_underlying_price < put_option.strike_price:
                    # if in the money, pool gets profit
                    profit = (put_option.strike_price - self.current_underlying_price)
                    self.put_pool_size += put_option.quantity * profit
                else:
                    # if out of moeny, pool gets nothing
                    self.put_pool_size += 0
            else:
                if self.current_underlying_price < put_option.strike_price:
                    # if in the money, pool gets only part of the locked capital
                    loss = (put_option.strike_price - self.current_underlying_price)
                    self.put_pool_size += put_option.locked_capital - put_option.quantity * loss
                else:
                    # if out of money
                    self.put_pool_size += put_option.locked_capital
            self._remove_option(put_option)

    def __dict__(self) -> Dict[str, Any]:
        return {
            'call_strikes': self.call_strikes,
            'put_strikes': self.put_strikes,
            'call_volatility': self.call_volatility,
            'put_volatility': self.put_volatility,
            'call_pool_size': self.call_pool_size,
            'put_pool_size': self.put_pool_size,
            'call_issued_options': self.call_issued_options,
            'put_issued_options': self.put_issued_options,
            'time_till_maturity': self.time_till_maturity,
            'current_underlying_price': self.current_underlying_price,
        }
