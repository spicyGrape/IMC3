from datamodel import Order, OrderDepth, Trade, TradingState
from typing import List, Dict
import json
import math
import numpy as np
import pandas as pd

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


class Trader:
    def __init__(self):
        self.price_history = {"SQUID_INK": [], "KELP": [], "RAINFOREST_RESIN": []}
        self.kelp_last_trade_time = -KELP_COOLDOWN

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        current_time = state.timestamp

        for product in state.order_depths:
            if product not in self.price_history:
                continue
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            best_bid = max(order_depth.buy_orders.keys(), default=None)
            best_ask = min(order_depth.sell_orders.keys(), default=None)
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None

            if mid_price:
                self.price_history[product].append(mid_price)
                if len(self.price_history[product]) > 30:
                    self.price_history[product] = self.price_history[product][-30:]

            prices = self.price_history[product]

            # RESIN 稳定套利策略
            if product == "RAINFOREST_RESIN":
                fair_price = 10000
                spread = 1
                if current_position < POSITION_LIMIT:
                    orders.append(
                        Order(
                            product,
                            fair_price - spread,
                            min(3, POSITION_LIMIT - current_position),
                        )
                    )
                if current_position > -POSITION_LIMIT:
                    orders.append(
                        Order(
                            product,
                            fair_price + spread,
                            -min(3, current_position + POSITION_LIMIT),
                        )
                    )
                result[product] = orders
                continue

            # SQUID RSI 策略 + 方向确认机制
            if product == "SQUID_INK":
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

                        if signal == "BUY" and best_ask:
                            orders.append(
                                Order(
                                    product,
                                    safe_price(best_ask),
                                    min(5, POSITION_LIMIT - current_position),
                                )
                            )
                        elif signal == "SELL" and best_bid:
                            orders.append(
                                Order(
                                    product,
                                    safe_price(best_bid),
                                    -min(5, current_position + POSITION_LIMIT),
                                )
                            )

                result[product] = orders
                continue

            # KELP 趋势确认策略（稳健版）
            if product == "KELP":
                if len(prices) >= 10 and (
                    current_time - self.kelp_last_trade_time >= KELP_COOLDOWN
                ):
                    slope = linear_trend_slope(prices[-10:])
                    recent_volatility = np.std(prices[-5:])

                    if (
                        abs(slope) > SLOPE_THRESHOLD
                        and recent_volatility > MIN_VOLATILITY
                    ):
                        direction = "BUY" if slope > 0 else "SELL"
                        volume = 1
                        if direction == "BUY" and best_ask:
                            orders.append(
                                Order(
                                    product,
                                    safe_price(best_ask),
                                    min(volume, POSITION_LIMIT - current_position),
                                )
                            )
                            self.kelp_last_trade_time = current_time
                        elif direction == "SELL" and best_bid:
                            orders.append(
                                Order(
                                    product,
                                    safe_price(best_bid),
                                    -min(volume, current_position + POSITION_LIMIT),
                                )
                            )
                            self.kelp_last_trade_time = current_time

                result[product] = orders

        return result, conversions, ""
