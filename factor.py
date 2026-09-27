"""Only labels use future prices. Labels are never supplied to the LLM."""
import pandas as pd
import config


def calculate_factor(prices):
    stocks = prices[config.TICKERS]
    momentum = stocks / stocks.shift(config.MOMENTUM_WINDOW) - 1
    future = stocks.shift(-config.FORWARD_WINDOW) / stocks - 1
    return momentum, future


def rank_ic(momentum, future):
    # Pairwise deletion BEFORE ranking is essential for a correct Spearman IC.
    valid = momentum.notna() & future.notna()
    x = momentum.where(valid).rank(axis=1)
    y = future.where(valid).rank(axis=1)
    ic = x.corrwith(y, axis=1).where(valid.sum(axis=1) >= config.MIN_STOCKS)
    return ic.rename('rank_ic')


def test_ic(prices):
    momentum, future = calculate_factor(prices)
    ic = rank_ic(momentum, future)
    label_end = pd.Series(prices.index, index=prices.index).shift(-config.FORWARD_WINDOW)
    end = pd.Timestamp(config.TEST_END) if config.TEST_END else prices.index.max()
    # Both the signal date and its entire forward label must lie in Test.
    eligible = (ic.index >= pd.Timestamp(config.TEST_START)) & (ic.index <= end) & (label_end <= end)
    return ic.loc[eligible].dropna()
