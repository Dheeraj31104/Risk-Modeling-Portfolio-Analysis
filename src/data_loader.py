import yfinance as yf
import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GS",
           "JNJ",  "PFE",  "XOM",   "CVX", "BND"]

CRISES = {
    "2022 rate hike": (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-12")),
    "SVB collapse":   (pd.Timestamp("2023-03-08"), pd.Timestamp("2023-03-31")),
}


def load_portfolio(tickers=TICKERS, period="5y"):
    """
    Fetch prices, compute log returns and equal-weight portfolio return.

    Returns
    -------
    prices   : DataFrame  (N, n_assets)
    log_ret  : DataFrame  (N-1, n_assets)
    port_ret : Series     (N-1,)   equal-weight portfolio log return
    weights  : ndarray    (n_assets,)
    """
    prices = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    prices.dropna(inplace=True)

    # yfinance can return tz-aware index — strip tz for matplotlib compatibility
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert(None)

    log_ret  = np.log(prices / prices.shift(1)).dropna()
    weights  = np.ones(len(tickers)) / len(tickers)
    port_ret = log_ret @ weights

    print(f"Loaded {len(prices)} trading days: "
          f"{prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Portfolio  mean: {port_ret.mean():.4f}  "
          f"std: {port_ret.std():.4f}  "
          f"skew: {port_ret.skew():.4f}  "
          f"kurt: {port_ret.kurt():.4f}")

    return prices, log_ret, port_ret, weights
