"""Freeze first, then evaluate. Existing experiments never silently regenerate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import config
from data import load_data
from evaluator import calibrate, evaluate, statistics
from factor import test_ic
from features import build_features
from hypothesis import parse_hypotheses, random_hypotheses
from llm import generate_hypotheses


def save_json(path, obj):
    # Exclusive creation deliberately refuses to overwrite frozen artifacts.
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_discovery_snapshot(path, prices, volumes):
    """Store the exact provider bars used at freeze time, including feature warmup."""
    symbols = prices.columns
    parts = [pd.DataFrame({'date': prices.index, 'ticker': symbol,
                           'price': prices[symbol].to_numpy(),
                           'volume': volumes[symbol].to_numpy()}) for symbol in symbols]
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)


def check_conditions(frozen, hypotheses, discovery, allow_vendor_revision=False):
    expected = calibrate(hypotheses, discovery)
    random_expected = calibrate(random_hypotheses(len(hypotheses), config.RANDOM_SEED), discovery)
    for label, original, current in [('hypothesis', frozen['conditions'], expected),
                                      ('random', frozen['random_conditions'], random_expected)]:
        if len(original) != len(current):
            raise ValueError(f'Frozen {label} condition count changed.')
        for old, new in zip(original, current):
            for key in new.keys() - {'threshold_value'}:
                if old.get(key) != new[key]:
                    raise ValueError(f'Frozen {label} condition definition was modified.')
            if allow_vendor_revision:
                if not np.isclose(old['threshold_value'], new['threshold_value'], rtol=1e-4, atol=1e-6):
                    raise ValueError(f'Historical data revision materially changed {label} threshold for {old["name"]}; use a separate results directory for a new experiment.')
            elif old['threshold_value'] != new['threshold_value']:
                raise ValueError('Frozen numeric conditions were modified. Restore the original frozen file.')


def protocol():
    if not (pd.Timestamp(config.DATA_START) <= pd.Timestamp(config.DISCOVERY_START)
            <= pd.Timestamp(config.DISCOVERY_END) < pd.Timestamp(config.TEST_START)):
        raise ValueError('Require DATA_START <= DISCOVERY_START <= DISCOVERY_END < TEST_START.')
    if config.TEST_END and pd.Timestamp(config.TEST_END) < pd.Timestamp(config.TEST_START):
        raise ValueError('TEST_END precedes TEST_START.')
    if not 3 <= config.MIN_STOCKS <= len(config.TICKERS):
        raise ValueError('MIN_STOCKS must be between 3 and the stock count.')
    keys = ['TICKERS', 'MARKET', 'DATA_START', 'DATA_END', 'DISCOVERY_START',
            'DISCOVERY_END', 'TEST_START', 'TEST_END', 'MOMENTUM_WINDOW',
            'FORWARD_WINDOW', 'MIN_STOCKS', 'RANDOM_SEED', 'HYPOTHESIS_COUNT']
    settings = {key: getattr(config, key) for key in keys}
    # Protect against changing feature/evaluation implementations between reruns.
    settings['code_hash'] = hashlib.sha256(b''.join(
        (config.ROOT / f'{name}.py').read_bytes() for name in
        ['config', 'data', 'features', 'factor', 'hypothesis', 'llm', 'evaluator', 'run'])).hexdigest()
    return settings


def run(args):
    settings = protocol()
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    hypothesis_path = out / 'hypotheses.json'
    frozen_path = out / 'frozen_conditions.json'
    snapshot_path = out / 'discovery_data.csv'
    if args.llm and (hypothesis_path.exists() or frozen_path.exists()):
        raise ValueError('Frozen hypotheses already exist. Rerun without --llm; never regenerate after viewing Test.')
    if hypothesis_path.exists():
        hypotheses = parse_hypotheses(hypothesis_path.read_text(), config.HYPOTHESIS_COUNT)
        if args.hypotheses:
            supplied = parse_hypotheses(Path(args.hypotheses).read_text(), config.HYPOTHESIS_COUNT)
            if supplied != hypotheses:
                raise ValueError('Provided hypotheses differ from frozen hypotheses.')
    else:
        if frozen_path.exists():
            raise ValueError('Frozen conditions exist but hypotheses.json is missing.')
        if args.llm:
            hypotheses = generate_hypotheses()
        else:
            source = Path(args.hypotheses) if args.hypotheses else config.ROOT / 'manual_hypotheses.json'
            hypotheses = parse_hypotheses(source.read_text(), config.HYPOTHESIS_COUNT)
        save_json(hypothesis_path, {'hypotheses': hypotheses})
    old = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    if old:
        old_settings = old['protocol']
        if ({key: value for key, value in old_settings.items() if key != 'code_hash'} !=
                {key: value for key, value in settings.items() if key != 'code_hash'} or
                old['hypotheses_hash'] != digest(hypotheses)):
            raise ValueError('Frozen protocol/hypotheses changed. Restore the original files; do not tune on Test.')
        if old_settings.get('code_hash') != settings['code_hash']:
            print('Note: source code changed since freeze; review before comparing results.', flush=True)
    print('Hypotheses frozen. Loading daily data...', flush=True)
    prices, volumes = load_data(args.csv)
    if old and old.get('snapshot_hash'):
        if not snapshot_path.exists():
            raise ValueError('Frozen Discovery snapshot is missing; restore discovery_data.csv.')
        if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != old['snapshot_hash']:
            raise ValueError('Frozen Discovery snapshot changed; restore discovery_data.csv.')
    if old and snapshot_path.exists():
        saved_prices, saved_volumes = load_data(snapshot_path)
        if saved_prices.index.max() > pd.Timestamp(config.DISCOVERY_END):
            raise ValueError('Discovery snapshot contains dates beyond Discovery end.')
        prices = pd.concat([saved_prices, prices.loc[prices.index > pd.Timestamp(config.DISCOVERY_END)]])
        volumes = pd.concat([saved_volumes, volumes.loc[volumes.index > pd.Timestamp(config.DISCOVERY_END)]])
    if prices.index.max() < pd.Timestamp(config.DISCOVERY_END):
        # Allow non-trading days at the end of the Discovery interval.
        if (pd.Timestamp(config.DISCOVERY_END) - prices.index.max()).days > 7:
            raise ValueError('Data does not cover the Discovery period.')
    # Calibrate on a physically sliced prefix, never on Test feature values.
    discovery_prices = prices.loc[:config.DISCOVERY_END]
    discovery_volumes = volumes.loc[:config.DISCOVERY_END]
    discovery = build_features(discovery_prices, discovery_volumes).loc[config.DISCOVERY_START:config.DISCOVERY_END]
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(discovery, index=True).values.tobytes()).hexdigest()
    if old:
        revised = old['discovery_hash'] != data_hash
        if revised and snapshot_path.exists():
            raise ValueError('Saved Discovery snapshot does not match frozen data; restore the original snapshot.')
        check_conditions(old, hypotheses, discovery, allow_vendor_revision=revised)
        if revised:
            print('Warning: provider revised Discovery data and this older experiment has no data snapshot. Original frozen thresholds are retained; this rerun is not byte-for-byte reproducible.', flush=True)
        frozen = old
    else:
        randoms = random_hypotheses(len(hypotheses), config.RANDOM_SEED)
        frozen = {'protocol': settings, 'hypotheses_hash': digest(hypotheses),
                  'discovery_hash': data_hash,
                  'source': 'llm' if args.llm else 'manual',
                  'data_source': str(Path(args.csv).resolve()) if args.csv else 'yfinance adjusted close',
                  'conditions': calibrate(hypotheses, discovery),
                  'random_conditions': calibrate(randoms, discovery)}
        save_discovery_snapshot(snapshot_path, discovery_prices, discovery_volumes)
        frozen['snapshot_hash'] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        save_json(frozen_path, frozen)
    print(f"Numeric thresholds frozen; source={frozen['source']}. Evaluating Test...", flush=True)
    features = build_features(prices, volumes)
    ic = test_ic(prices)
    if ic.empty:
        raise ValueError('No valid Test IC observations; check dates, coverage and MIN_STOCKS.')
    result = evaluate(frozen['conditions'], features, ic)
    random_result = evaluate(frozen['random_conditions'], features, ic)
    result.to_csv(out / 'evaluation.csv', index=False)
    random_result.to_csv(out / 'random_evaluation.csv', index=False)
    ic.to_csv(out / 'daily_ic.csv')
    baseline = statistics(ic)
    print(f"\nBaseline Test Rank IC: {baseline['ic']:.6f}")
    print(f"IC std: {baseline['ic_std']:.6f} | naive t-stat: {baseline['ic_t_stat_naive']:.3f}")
    print(f"Positive IC ratio: {baseline['positive_ic_ratio']:.3f} | sample days: {baseline['days']}")
    print(f"Signal dates: {ic.index.min().date()} to {ic.index.max().date()}")
    for i, row in enumerate(result.itertuples(), 1):
        print(f'\nHypothesis {i}: {row.name}')
        print(f'  Condition: {row.variable} {row.operator} {row.threshold_value:.6g} (Discovery P{row.threshold_percentile:g})')
        print(f'  Counterexample IC: {row.counterexample_ic:.6f}')
        print(f'  Non-counterexample IC: {row.normal_ic:.6f}')
        print(f'  Delta IC: {row.delta_ic:.6f}')
        print(f'  Counterexample Days: {row.counterexample_days} | Normal Days: {row.normal_days} | Missing: {row.missing_condition_days}')
    distribution = pd.DataFrame({frozen['source']: result.delta_ic, 'random': random_result.delta_ic}).describe()
    distribution.to_csv(out / 'delta_summary.csv')
    print('\nDelta IC distributions (descriptive only):\n' + distribution.to_string())
    print('\nNaive t-statistics ignore overlapping returns and serial dependence. No significance or profitability conclusion.')
    print(f'Results: {out.resolve()}')
    return result, random_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', help='Long-form adjusted daily prices: date,ticker,price,volume')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--hypotheses', help='Manual JSON file, or an existing results/hypotheses.json')
    source.add_argument('--llm', action='store_true', help='Generate once using LLM environment variables')
    parser.add_argument('--results-dir', default=str(config.RESULTS_DIR))
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
