import pandas as pd
import matplotlib.pyplot as plt

price_files = {
    "day_-2": "prices_round_1_day_-2.csv",
    "day_-1": "prices_round_1_day_-1.csv",
    "day_0": "prices_round_1_day_0.csv"
}

trade_files = {
    "day_-2": "trades_round_1_day_-2.csv",
    "day_-1": "trades_round_1_day_-1.csv",
    "day_0": "trades_round_1_day_0.csv"
}


def load_data(price_paths, trade_paths):
    price_dfs = []
    trade_dfs = []
    for day, path in price_paths.items():
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        price_dfs.append(df)
    for day, path in trade_paths.items():
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        trade_dfs.append(df)
    return pd.concat(price_dfs), pd.concat(trade_dfs)

prices_all, trades_all = load_data(price_files, trade_files)


products = prices_all["product"].unique()


def backtest_product(product: str):
    df = prices_all[prices_all["product"] == product].sort_values(by=["day", "timestamp"])
    df = df.dropna(subset=["mid_price"])

    
    WMA_WINDOW = 5
    pnl = []
    price_buffer = []

    for _, row in df.iterrows():
        price = row["mid_price"]
        price_buffer.append(price)
        if len(price_buffer) > WMA_WINDOW:
            price_buffer = price_buffer[-WMA_WINDOW:]

        if len(price_buffer) >= WMA_WINDOW:
            weights = list(range(1, WMA_WINDOW + 1))
            wma = sum(w * p for w, p in zip(weights, price_buffer)) / sum(weights)
            # 策略逻辑
            action = "HOLD"
            pnl_change = 0
            if price > wma:
                action = "SELL"
                pnl_change = price - wma
            elif price < wma:
                action = "BUY"
                pnl_change = wma - price
            else:
                action = "HOLD"
            pnl.append(pnl[-1] + pnl_change if pnl else pnl_change)
        else:
            pnl.append(0)

    df = df.iloc[:len(pnl)].copy()
    df["pnl"] = pnl
    return df


def plot_all():
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 4 * len(products)), sharex=True)
    if len(products) == 1:
        axes = [axes]

    for i, product in enumerate(products):
        bt = backtest_product(product)
        axes[i].plot(bt["timestamp"], bt["pnl"], label=f"{product} PnL")
        axes[i].set_title(f"{product} backtest profits")
        axes[i].set_ylabel("total PnL")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.show()


plot_all()
