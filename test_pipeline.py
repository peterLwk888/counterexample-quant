"""Offline tests for leakage, schema, IC and a complete CSV experiment."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import config
from data import load_data
from evaluator import calibrate, evaluate, statistics
from factor import calculate_factor, rank_ic, test_ic
from features import build_features
from hypothesis import parse_hypotheses, random_hypotheses
from llm import generate_hypotheses
from run import run


def fixture():
    rng = np.random.default_rng(123)
    dates = pd.bdate_range('2015-01-01', '2024-12-31')
    columns = config.TICKERS + [config.MARKET]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.0001, .012, (len(dates), len(columns))), axis=0)), index=dates, columns=columns)
    volumes = pd.DataFrame(rng.integers(1000, 100000, prices.shape), index=dates, columns=columns)
    return prices, volumes


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices, cls.volumes = fixture()
        cls.hypotheses = parse_hypotheses((config.ROOT / 'manual_hypotheses.json').read_text(), 5)

    def test_future_perturbation_and_prefix_invariance(self):
        original = build_features(self.prices, self.volumes)
        prices, volumes = self.prices.copy(), self.volumes.copy()
        prices.loc[config.TEST_START:] *= np.linspace(1, 100, len(prices.loc[config.TEST_START:]))[:, None]
        volumes.loc[config.TEST_START:] *= 100
        changed = build_features(prices, volumes)
        assert_frame_equal(original.loc[:config.DISCOVERY_END], changed.loc[:config.DISCOVERY_END])
        self.assertEqual(calibrate(self.hypotheses, original.loc[:config.DISCOVERY_END]), calibrate(self.hypotheses, changed.loc[:config.DISCOVERY_END]))
        cutoff = self.prices.index[200]
        assert_frame_equal(original.loc[:cutoff], build_features(self.prices.loc[:cutoff], self.volumes.loc[:cutoff]))

    def test_formulas_and_forward_boundary(self):
        momentum, future = calculate_factor(self.prices)
        self.assertAlmostEqual(momentum.iloc[20, 0], self.prices.iloc[20, 0] / self.prices.iloc[0, 0] - 1)
        self.assertAlmostEqual(future.iloc[20, 0], self.prices.iloc[25, 0] / self.prices.iloc[20, 0] - 1)
        self.assertTrue(future.tail(5).isna().all().all())
        with patch.object(config, 'TEST_END', '2023-06-30'):
            ic = test_ic(self.prices)
            ends = pd.Series(self.prices.index, index=self.prices.index).shift(-5)
            self.assertTrue((ends.loc[ic.index] <= pd.Timestamp('2023-06-30')).all())
            self.assertTrue((ic.index >= pd.Timestamp(config.TEST_START)).all())

    def test_spearman_pairwise_missing_ties_and_constant(self):
        x = pd.DataFrame([[1, 2, 3, 4], [1, 1, 3, 4], [1, 1, 1, 1]])
        y = pd.DataFrame([[4, np.nan, 3, 1], [1, 1, 3, 4], [1, 2, 3, 4]])
        with patch.object(config, 'MIN_STOCKS', 3), np.errstate(invalid='ignore', divide='ignore'):
            values = rank_ic(x, y)
        self.assertAlmostEqual(values.iloc[0], -1)
        self.assertAlmostEqual(values.iloc[1], 1)
        self.assertTrue(np.isnan(values.iloc[2]))

    def test_invalid_json_schema(self):
        for field, value in [('variable', 'tomorrow_return'), ('operator', '=='), ('threshold_type', 'absolute'), ('threshold', '90'), ('threshold', True), ('threshold', 101), ('economic_reason', 'COVID in 2020'), ('unexpected', 1)]:
            bad = dict(self.hypotheses[0]); bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                parse_hypotheses(json.dumps({'hypotheses': [bad]}))
        for text in ['```json\n{}\n```', '{"hypotheses": [], "hypotheses": []}', '{"hypotheses": []}']:
            with self.assertRaises(ValueError):
                parse_hypotheses(text)

    def test_missing_conditions_and_empty_subsets(self):
        ic = pd.Series([.2, -.1, .3], index=pd.date_range('2023-01-01', periods=3))
        features = pd.DataFrame({'market_volatility_20': [np.nan, 1, 2]}, index=ic.index)
        row = evaluate([dict(self.hypotheses[0], threshold_value=5)], features, ic).iloc[0]
        self.assertEqual(row.missing_condition_days, 1)
        self.assertEqual(row.counterexample_days, 0)
        self.assertEqual(row.normal_days, 2)
        self.assertTrue(np.isnan(row.delta_ic))
        self.assertAlmostEqual(statistics(ic)['ic'], .4 / 3)
        self.assertTrue(np.isnan(statistics(pd.Series([.1]))['ic_t_stat_naive']))

    def test_random_reproducibility(self):
        self.assertEqual(random_hypotheses(5, 42), random_hypotheses(5, 42))
        self.assertNotEqual(random_hypotheses(5, 42), random_hypotheses(5, 43))

    def test_mock_api_contract(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'hypotheses': self.hypotheses})}}]}
        with patch.dict('os.environ', {'LLM_MODEL': 'test-model', 'LLM_BASE_URL': 'http://localhost:8000/v1'}), patch('urllib.request.urlopen') as mocked:
            mocked.return_value.__enter__.return_value = io.StringIO(json.dumps(response))
            self.assertEqual(generate_hypotheses(), self.hypotheses)
            payload = json.loads(mocked.call_args.args[0].data)
            self.assertEqual(payload['model'], 'test-model')
            self.assertNotIn('baseline_ic', json.dumps(payload))
            self.assertNotIn('2022', json.dumps(payload))

    def test_complete_csv_and_freeze_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            frames = [pd.DataFrame({'date': self.prices.index, 'ticker': t, 'price': self.prices[t].values, 'volume': self.volumes[t].values}) for t in self.prices.columns]
            csv = path / 'synthetic.csv'
            pd.concat(frames).to_csv(csv, index=False)
            args = argparse.Namespace(csv=str(csv), results_dir=str(path / 'results'), llm=False, hypotheses=None)
            with contextlib.redirect_stdout(io.StringIO()):
                first, random = run(args)
                frozen = (path / 'results/frozen_conditions.json').read_bytes()
                second, _ = run(args)
            assert_frame_equal(first, second)
            self.assertEqual(len(random), 5)
            self.assertEqual(frozen, (path / 'results/frozen_conditions.json').read_bytes())
            self.assertTrue((path / 'results/evaluation.csv').exists())
            with patch.object(config, 'RANDOM_SEED', 999), self.assertRaisesRegex(ValueError, 'Frozen protocol'):
                run(args)
            (path / 'results/discovery_data.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'snapshot is missing'):
                run(args)
            legacy = json.loads(frozen)
            legacy.pop('snapshot_hash')
            frozen = json.dumps(legacy).encode()
            (path / 'results/frozen_conditions.json').write_bytes(frozen)
            revised_prices = self.prices.copy()
            revised_prices.loc['2018-01-02', 'SPY'] *= 1.000000001
            output = io.StringIO()
            with patch('run.load_data', return_value=(revised_prices, self.volumes)), contextlib.redirect_stdout(output):
                revised, _ = run(args)
            self.assertIn('provider revised Discovery data', output.getvalue())
            self.assertEqual(frozen, (path / 'results/frozen_conditions.json').read_bytes())
            self.assertEqual(first.threshold_value.tolist(), revised.threshold_value.tolist())
            altered = json.loads(frozen)
            altered['conditions'][0]['threshold_value'] += 1
            (path / 'results/frozen_conditions.json').write_text(json.dumps(altered))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'numeric conditions'):
                run(args)
            raw = pd.read_csv(csv)
            pd.concat([raw, raw.iloc[:1]]).to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_data(csv)


if __name__ == '__main__':
    unittest.main()
