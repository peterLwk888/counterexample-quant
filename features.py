"""Market states available after close t; rolling windows include t."""
import numpy as np
import pandas as pd
import config

VARIABLE_DEFINITIONS = {
    'market_return_5': 'SPY trailing 5-session adjusted close return',
    'market_return_20': 'SPY trailing 20-session adjusted close return',
    'market_volatility_20': 'Sample std of SPY last 20 daily returns times sqrt(252)',
    'market_drawdown_60': 'SPY adjusted close / trailing 60-session maximum - 1',
    'market_volume_zscore_20': '(SPY volume - trailing 20-session mean) / sample std',
    'cross_sectional_dispersion': 'Sample std of same-day stock returns; excludes SPY',
    'momentum_dispersion': 'Sample std of stock trailing 20-session returns; excludes SPY',
}


def build_features(prices, volumes):
    market = prices[config.MARKET]
    returns = prices[config.TICKERS].pct_change(fill_method=None)
    momentum = prices[config.TICKERS].pct_change(config.MOMENTUM_WINDOW, fill_method=None)
    volume = volumes[config.MARKET]
    result = pd.DataFrame({
        'market_return_5': market.pct_change(5, fill_method=None),
        'market_return_20': market.pct_change(20, fill_method=None),
        'market_volatility_20': market.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252),
        'market_drawdown_60': market / market.rolling(60).max() - 1,
        'market_volume_zscore_20': (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan),
        'cross_sectional_dispersion': returns.std(axis=1).where(returns.count(axis=1) >= config.MIN_STOCKS),
        'momentum_dispersion': momentum.std(axis=1).where(momentum.count(axis=1) >= config.MIN_STOCKS),
    })
    return result.replace([np.inf, -np.inf], np.nan)
