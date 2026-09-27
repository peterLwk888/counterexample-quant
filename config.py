"""Small, explicit research configuration. Date bounds are inclusive."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'XOM',
           'JNJ', 'WMT', 'PG', 'HD', 'BAC', 'KO', 'PEP', 'CVX', 'CSCO',
           'DIS', 'MCD', 'IBM']
MARKET = 'SPY'
DATA_START = '2015-01-01'
DATA_END = None  # Latest completed daily bar; current New York day is excluded.
DISCOVERY_START = '2015-01-01'
DISCOVERY_END = '2021-12-31'
TEST_START = '2022-01-01'
TEST_END = None
MOMENTUM_WINDOW = 20
FORWARD_WINDOW = 5
MIN_STOCKS = 10
RANDOM_SEED = 42
HYPOTHESIS_COUNT = 5
RESULTS_DIR = ROOT / 'results'
