from datamodel import Order, OrderDepth, Trade, TradingState
from typing import List, Dict
import json
import math
import numpy as np
import pandas as pd
import os

POSITION_LIMIT = 20
WMA_WINDOWS = {"SQUID_INK": 5, "KELP": 5}
IDLE_THRESHOLD = 5
MIN_VOLATILITY = 1.2
MAX_ORDER_VOLUME = 10
SLOPE_THRESHOLD = 0.15
KELP_COOLDOWN = 5


def safe_price(p):
    return int(round(p))


def compute_rsi(prices: pd.Series, window: int = 6) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def linear_trend_slope(prices: list) -> float:
    if len(prices) < 5:
        return 0.0
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope


class MarketData:
    def __init__(self, order_depth: OrderDepth):
        self.order_depth = order_depth
        self.best_bid = max(order_depth.buy_orders.keys(), default=None)
        self.best_ask = min(order_depth.sell_orders.keys(), default=None)
        self.mid_price = (
            (self.best_bid + self.best_ask) / 2
            if self.best_bid and self.best_ask
            else None
        )
        self.bid_volume = sum(order_depth.buy_orders.values())
        self.ask_volume = -sum(order_depth.sell_orders.values())
        self.spread = (
            (self.best_ask - self.best_bid) if self.best_bid and self.best_ask else None
        )


class TradingStrategy:
    def __init__(self, product, position_limit=20):
        self.product = product
        self.position_limit = position_limit
        self.price_history = []

    def update_price_history(self, mid_price):
        if mid_price:
            self.price_history.append(mid_price)
            if len(self.price_history) > 30:  # Keep reasonable history
                self.price_history = self.price_history[-30:]

    def create_buy_order(self, price, market_data, current_position, max_volume=None):
        """Helper method to create a buy order with position limit checks"""
        if current_position >= self.position_limit:
            return None

        if max_volume is None:
            max_volume = self.position_limit - current_position

        return Order(
            self.product,
            safe_price(price),
            min(max_volume, self.position_limit - current_position),
        )

    def create_sell_order(self, price, market_data, current_position, max_volume=None):
        """Helper method to create a sell order with position limit checks"""
        if current_position <= -self.position_limit:
            return None

        if max_volume is None:
            max_volume = current_position + self.position_limit

        return Order(
            self.product,
            safe_price(price),
            -min(max_volume, current_position + self.position_limit),
        )

    def generate_orders(self, market_data, current_position, timestamp):
        """To be implemented by subclasses"""
        raise NotImplementedError


class ResinStrategy(TradingStrategy):
    def generate_orders(self, market_data, current_position, timestamp) -> list[Order]:
        orders = []
        prices = self.price_history

        if len(prices) < 10:
            anchor = 10000
        else:
            anchor = int(np.median(prices[-10:]))

        spread = (
            market_data.spread if market_data.spread and market_data.spread >= 2 else 2
        )

        buy_price = anchor - spread // 2
        sell_price = anchor + spread // 2

        buy_order = self.create_buy_order(buy_price, market_data, current_position, 3)
        if buy_order:
            orders.append(buy_order)
        sell_order = self.create_sell_order(
            sell_price, market_data, current_position, 3
        )
        if sell_order:
            orders.append(sell_order)

        return orders


class SquidInkStrategy(TradingStrategy):
    def __init__(self, product, position_limit=POSITION_LIMIT):
        super().__init__(product, position_limit)

    def generate_orders(self, market_data, current_position, timestamp) -> list[Order]:
        orders = []
        prices = self.price_history
        if len(prices) >= 10:
            df = pd.DataFrame({"mid_price": prices})
            df["rsi"] = compute_rsi(df["mid_price"], window=6)
            df["momentum"] = df["mid_price"].diff(3)
            df = df.dropna()
            if not df.empty:
                latest = df.iloc[-1]
                rsi = latest["rsi"]
                momentum = latest["momentum"]
                slope = linear_trend_slope(prices[-10:])
                signal = "HOLD"
                if rsi < 40 and momentum > 0 and slope > 0:
                    signal = "BUY"
                elif rsi > 60 and momentum < 0 and slope < 0:
                    signal = "SELL"
                if signal == "BUY" and market_data.best_ask:
                    if market_data.ask_volume > 2 * market_data.bid_volume:
                        buy_order = self.create_buy_order(
                            market_data.best_ask, market_data, current_position, 6
                        )
                        if buy_order:
                            orders.append(buy_order)
                elif signal == "SELL" and market_data.best_bid:
                    if market_data.bid_volume > 2 * market_data.ask_volume:
                        sell_order = self.create_sell_order(
                            market_data.best_bid, market_data, current_position, 6
                        )
                        if sell_order:
                            orders.append(sell_order)
        return orders


class KelpStrategy(TradingStrategy):
    def __init__(self, product, position_limit=POSITION_LIMIT):
        super().__init__(product, position_limit)
        self.cooldown = KELP_COOLDOWN
        self.last_trade_time = -self.cooldown

    def compute_ma(self, window):
        if len(self.price_history) < window:
            return np.mean(self.price_history)
        return np.mean(self.price_history[-window:])

    def generate_orders(self, market_data, current_position, timestamp) -> list[Order]:
        orders = []
        if market_data.mid_price is None:
            return orders

        if timestamp - self.last_trade_time < self.cooldown:
            return orders

        ma_short = self.compute_ma(3)
        ma_long = self.compute_ma(6)
        mid_price = market_data.mid_price

        if ma_short > ma_long and current_position < self.position_limit:
            buy_order = self.create_buy_order(
                mid_price - 1, market_data, current_position, 2
            )
            if buy_order:
                orders.append(buy_order)
                self.last_trade_time = timestamp

        elif ma_short < ma_long and current_position > -self.position_limit:
            sell_order = self.create_sell_order(
                mid_price + 1, market_data, current_position, 2
            )
            if sell_order:
                orders.append(sell_order)
                self.last_trade_time = timestamp

        return orders


class PicnicBasket1Strategy(TradingStrategy):
    spread_backtrace = []
    WINDOW_SIZE = 1000
    UPPER_THRESHOLD = float(os.getenv("UPPER_THRESHOLD", 1.05))
    LOWER_THRESHOLD = -UPPER_THRESHOLD
    
    PRICE_MARGIN_THRESHOLD = int(os.getenv("PICNIC_BASKET1_PRICE_MARGIN_THRESHOLD", 200))

    def __init__(self, product, position_limit=60):
        super().__init__(product, position_limit)

    def generate_orders(self, market_data, current_position, timestamp, state):
        """TODO: Implement policy to ensure short amount matches long amount"""

        orders = []

        # Collect market data for all components
        basket1_data = MarketData(state.order_depths["PICNIC_BASKET1"])
        croissants_data = MarketData(state.order_depths["CROISSANTS"])
        jams_data = MarketData(state.order_depths["JAMS"])
        djembe_data = MarketData(state.order_depths["DJEMBES"])

        # one basket1 contains 6 croissants, 3 jams, and 1 djembe
        estimate_basket1_price = (
            6 * croissants_data.mid_price
            + 3 * jams_data.mid_price
            + djembe_data.mid_price
        )
        
        spread = basket1_data.best_bid - estimate_basket1_price
        self.spread_backtrace.append(spread)
        if len(self.spread_backtrace) < self.WINDOW_SIZE:
            return orders
        mean = np.mean(self.spread_backtrace[-self.WINDOW_SIZE:])
        std = np.std(self.spread_backtrace[-self.WINDOW_SIZE:])
        z_score = (spread - mean) / std

        # When people overpay for basket1, we can sell it and buy the components
        if z_score > self.UPPER_THRESHOLD:
            sell_order: None | Order = self.create_sell_order(
                basket1_data.best_bid, market_data, current_position, 1
            )
            if sell_order:
                # Create buy orders for the components
                buy_order_croissants: None | Order = self.create_buy_order(
                    croissants_data.best_ask, market_data, current_position, 6
                )
                buy_order_jams: None | Order = self.create_buy_order(
                    jams_data.best_ask, market_data, current_position, 3
                )
                buy_order_djembe: None | Order = self.create_buy_order(
                    djembe_data.best_ask, market_data, current_position, 1
                )
                # Check if all buy orders are valid
                if buy_order_croissants and buy_order_jams and buy_order_djembe:
                    orders.append(sell_order)
                    orders.append(buy_order_croissants)
                    orders.append(buy_order_jams)
                    orders.append(buy_order_djembe)
        # When people underpay for basket1, we can buy it and sell the components
        elif (
            z_score < self.LOWER_THRESHOLD
        ):
            buy_order: None | Order = self.create_buy_order(
                basket1_data.best_ask, market_data, current_position, 1
            )
            if buy_order:
                # Create sell orders for the components
                sell_order_croissants: None | Order = self.create_sell_order(
                    croissants_data.best_bid, market_data, current_position, 6
                )
                sell_order_jams: None | Order = self.create_sell_order(
                    jams_data.best_bid, market_data, current_position, 3
                )
                sell_order_djembe: None | Order = self.create_sell_order(
                    djembe_data.best_bid, market_data, current_position, 1
                )
                # Check if all sell orders are valid
                if sell_order_croissants and sell_order_jams and sell_order_djembe:
                    orders.append(buy_order)
                    orders.append(sell_order_croissants)
                    orders.append(sell_order_jams)
                    orders.append(sell_order_djembe)
        return orders

class PicnicBasket2Strategy(TradingStrategy):
    PRICE_MARGIN_THRESHOLD2 = int(os.getenv("PICNIC_BASKET2_PRICE_MARGIN_THRESHOLD2", 100))

    def __init__(self, product, position_limit=60):
        super().__init__(product, position_limit)

    def generate_orders(self, market_data, current_position, timestamp, state):
        """Implement policy to ensure short amount matches long amount for Basket2"""

        orders = []

        # Collect market data for all components
        basket2_data = MarketData(state.order_depths["PICNIC_BASKET2"])
        croissants_data = MarketData(state.order_depths["CROISSANTS"])
        jams_data = MarketData(state.order_depths["JAMS"])

        # one basket2 contains 4 croissants and 2 jams
        estimate_basket2_price = (
            4 * croissants_data.mid_price + 2 * jams_data.mid_price
        )

        # When people overpay for basket2, sell it and buy the components
        if basket2_data.best_bid - estimate_basket2_price > self.PRICE_MARGIN_THRESHOLD2:
            sell_order: None | Order = self.create_sell_order(
                basket2_data.best_bid, market_data, current_position, 1
            )
            if sell_order:
                buy_order_croissants: None | Order = self.create_buy_order(
                    croissants_data.best_ask, market_data, current_position, 4
                )
                buy_order_jams: None | Order = self.create_buy_order(
                    jams_data.best_ask, market_data, current_position, 2
                )
                if buy_order_croissants and buy_order_jams:
                    orders.append(sell_order)
                    orders.append(buy_order_croissants)
                    orders.append(buy_order_jams)

        # When people underpay for basket2, buy it and sell the components
        elif estimate_basket2_price - basket2_data.best_ask > self.PRICE_MARGIN_THRESHOLD2:
            buy_order: None | Order = self.create_buy_order(
                basket2_data.best_ask, market_data, current_position, 1
            )
            if buy_order:
                sell_order_croissants: None | Order = self.create_sell_order(
                    croissants_data.best_bid, market_data, current_position, 4
                )
                sell_order_jams: None | Order = self.create_sell_order(
                    jams_data.best_bid, market_data, current_position, 2
                )
                if sell_order_croissants and sell_order_jams:
                    orders.append(buy_order)
                    orders.append(sell_order_croissants)
                    orders.append(sell_order_jams)
        return orders

class Trader:
    def __init__(self):
        # Initialize strategies for each product
        self.r1_strategies: dict[str, TradingStrategy] = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN"),
            "SQUID_INK": SquidInkStrategy("SQUID_INK"),
            "KELP": KelpStrategy("KELP"),
        }
        self.r2_strategies: dict[str, TradingStrategy] = {
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1"),
            # "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2"),
        }

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        for product in state.order_depths:
            if product in self.r1_strategies:
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)

                # Create market data object with all calculations done once
                market_data = MarketData(order_depth)

                # Update strategy with new price
                self.r1_strategies[product].update_price_history(market_data.mid_price)

                # Generate orders using the appropriate strategy
                orders = self.r1_strategies[product].generate_orders(
                    market_data, current_position, state.timestamp
                )
                result[product] = orders
            elif product in self.r2_strategies:
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)

                # Generate orders using the appropriate strategy
                orders = self.r2_strategies[product].generate_orders(
                    market_data, current_position, state.timestamp, state=state
                )
                result[product] = orders

        return result, conversions, ""
