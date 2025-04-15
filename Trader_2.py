from datamodel import Order, OrderDepth, Trade, TradingState
from typing import List, Dict
import json
import math
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
import collections
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

POSITION_LIMIT = 50
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
    def __init__(self, product, position_limit=POSITION_LIMIT):
        self.product = product
        self.position_limit = position_limit
        self.price_history = []

    def update_price_history(self, mid_price):
        if mid_price:
            self.price_history.append(mid_price)
            if len(self.price_history) > 30:
                self.price_history = self.price_history[-30:]

    def create_buy_order(self, price, market_data, current_position, max_volume=None):
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

        if timestamp - self.last_trade_time < KELP_COOLDOWN:
            return orders

        slope = linear_trend_slope(prices[-10:])
        rsi = compute_rsi(pd.Series(prices), window=6).iloc[-1]

        signal = "HOLD"
        if slope > SLOPE_THRESHOLD and rsi < 65:
            signal = "BUY"
        elif slope < -SLOPE_THRESHOLD and rsi > 35:
            signal = "SELL"

        mid_price = int(round((market_data.best_bid + market_data.best_ask) / 2))

        if signal == "BUY":
            buy_order = self.create_buy_order(
                mid_price - 1, market_data, current_position, 2
            )
            if buy_order:
                orders.append(buy_order)
                self.last_trade_time = timestamp
        elif signal == "SELL":
            sell_order = self.create_sell_order(
                mid_price + 1, market_data, current_position, 2
            )
            if sell_order:
                orders.append(sell_order)
                self.last_trade_time = timestamp

        return orders

        slope = linear_trend_slope(prices[-10:])
        direction = "UP" if slope > 0.1 else "DOWN" if slope < -0.1 else "NEUTRAL"

        if direction == "UP" and market_data.best_ask:
            buy_order = self.create_buy_order(
                market_data.best_ask, market_data, current_position, 2
            )
            if buy_order:
                orders.append(buy_order)
        elif direction == "DOWN" and market_data.best_bid:
            sell_order = self.create_sell_order(
                market_data.best_bid, market_data, current_position, 2
            )
            if sell_order:
                orders.append(sell_order)

        return orders


class PicnicBasket1Strategy(TradingStrategy):
    PRICE_MARGIN_THRESHOLD = 200

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

        # When people overpay for basket1, we can sell it and buy the components
        if basket1_data.best_bid - estimate_basket1_price > self.PRICE_MARGIN_THRESHOLD:
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
            estimate_basket1_price - basket1_data.best_ask > self.PRICE_MARGIN_THRESHOLD
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


class JamStrategy(TradingStrategy):
    def __init__(
        self, product, position_limit=POSITION_LIMIT, window=10, spread_min=2, volume=3
    ):
        super().__init__(product, position_limit)
        self.window = window
        self.spread_min = spread_min
        self.volume = volume

    def generate_orders(self, market_data, current_position, timestamp):
        orders = []
        self.update_price_history(market_data.mid_price)

        if len(self.price_history) < self.window or market_data.mid_price is None:
            return orders

        anchor = int(np.median(self.price_history[-self.window :]))
        spread = (
            market_data.spread
            if market_data.spread and market_data.spread >= self.spread_min
            else self.spread_min
        )

        buy_price = anchor - spread // 2
        sell_price = anchor + spread // 2

        buy_order = self.create_buy_order(
            buy_price, market_data, current_position, self.volume
        )
        if buy_order:
            orders.append(buy_order)

        sell_order = self.create_sell_order(
            sell_price, market_data, current_position, self.volume
        )
        if sell_order:
            orders.append(sell_order)

        return orders


class CroissantStrategy(TradingStrategy):
    def __init__(
        self, product, position_limit=POSITION_LIMIT, window=10, spread_min=2, volume=3
    ):
        super().__init__(product, position_limit)
        self.window = window
        self.spread_min = spread_min
        self.volume = volume

    def generate_orders(self, market_data, current_position, timestamp):
        orders = []
        self.update_price_history(market_data.mid_price)

        if len(self.price_history) < self.window or market_data.mid_price is None:
            return orders

        anchor = int(np.median(self.price_history[-self.window :]))
        spread = (
            market_data.spread
            if market_data.spread and market_data.spread >= self.spread_min
            else self.spread_min
        )

        buy_price = anchor - spread // 2
        sell_price = anchor + spread // 2

        buy_order = self.create_buy_order(
            buy_price, market_data, current_position, self.volume
        )
        if buy_order:
            orders.append(buy_order)

        sell_order = self.create_sell_order(
            sell_price, market_data, current_position, self.volume
        )
        if sell_order:
            orders.append(sell_order)

        return orders


class Trader:
    def __init__(self):
        self.r1_strategies: dict[str, TradingStrategy] = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN"),
            "SQUID_INK": SquidInkStrategy("SQUID_INK"),
            "KELP": KelpStrategy("KELP"),
            "CROISSANTS": CroissantStrategy("CROISSANTS", position_limit=250),
            "JAMS": JamStrategy("JAMS", position_limit=350),
            # "DJEMBE": strategy removed
        }
        self.r2_strategies: dict[str, TradingStrategy] = {
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1"),
        }

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        for product in state.order_depths:
            if product in self.r1_strategies:
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)
                market_data = MarketData(order_depth)
                self.r1_strategies[product].update_price_history(market_data.mid_price)
                orders = self.r1_strategies[product].generate_orders(
                    market_data, current_position, state.timestamp
                )
                result[product] = orders
            elif product in self.r2_strategies:
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)
                market_data = MarketData(order_depth)
                orders = self.r2_strategies[product].generate_orders(
                    market_data, current_position, state.timestamp, state=state
                )
                result[product] = orders

        logger.flush(state, result, conversions, "")
        return result, conversions, ""
