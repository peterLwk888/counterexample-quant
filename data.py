"""Providers return the same long table; no imputation or silent fallback."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import config


def load_data(csv_path=None):
    symbols = config.TICKERS + [config.MARKET]
    if len(set(symbols)) != len(symbols):
        raise ValueError('Stock symbols must be unique and exclude the market symbol.')
    yesterday = pd.Timestamp(datetime.now(ZoneInfo('America/New_York')).date() - timedelta(days=1))
    end = min(pd.Timestamp(config.DATA_END), yesterday) if config.DATA_END else yesterday
    if csv_path:
        # CSV price must be consistently adjusted across all dates/symbols.
        raw = pd.read_csv(csv_path)
    else:
        import yfinance as yf
        yf.set_tz_cache_location(str(config.ROOT / '.cache' / 'yfinance'))
        downloaded = yf.download(symbols, start=config.DATA_START,
                                 end=(end.to_pydatetime() + timedelta(days=1)).strftime('%Y-%m-%d'),
                                 auto_adjust=True, progress=False, threads=False)
        if downloaded.empty:
            raise ValueError('yfinance returned no data. Supply --csv with date,ticker,price,volume.')
        frames = []
        for symbol in symbols:
            try:
                frames.append(pd.DataFrame({'date': downloaded.index,
                    'ticker': symbol, 'price': downloaded['Close'][symbol].to_numpy(),
                    'volume': downloaded['Volume'][symbol].to_numpy()}))
            except KeyError as exc:
                raise ValueError(f'Missing yfinance data for {symbol}') from exc
        raw = pd.concat(frames, ignore_index=True)
    required = {'date', 'ticker', 'price', 'volume'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Data requires columns: {sorted(required)}')
    raw['date'] = pd.to_datetime(raw['date'], errors='raise').dt.tz_localize(None).dt.normalize()
    raw = raw[raw.ticker.isin(symbols) & raw.date.between(config.DATA_START, end)].copy()
    if raw.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate date/ticker observations.')
    for column in ['price', 'volume']:
        raw[column] = pd.to_numeric(raw[column], errors='raise')
        if np.isinf(raw[column]).any():
            raise ValueError(f'Infinite {column}.')
    if (raw.price.dropna() <= 0).any() or (raw.volume.dropna() < 0).any():
        raise ValueError('Prices must be positive and volumes nonnegative.')
    prices = raw.pivot(index='date', columns='ticker', values='price').reindex(columns=symbols).sort_index()
    volumes = raw.pivot(index='date', columns='ticker', values='volume').reindex(columns=symbols).sort_index()
    if prices.empty or prices.notna().sum().eq(0).any():
        raise ValueError('At least one configured symbol has no prices.')
    calendar = prices.index[prices[config.MARKET].notna()]
    if volumes[config.MARKET].reindex(calendar).notna().sum() == 0:
        raise ValueError('Market volume is missing.')
    return prices.reindex(calendar), volumes.reindex(calendar)
