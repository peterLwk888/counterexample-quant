"""Deterministic, descriptive evaluation; no selection based on test results."""
import operator
import numpy as np
import pandas as pd

OPERATORS = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le}


def statistics(ic):
    values = ic.dropna()
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    std = float(values.std(ddof=1)) if n > 1 else np.nan
    return {'ic': mean, 'ic_std': std,
            'ic_t_stat_naive': mean / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan,
            'positive_ic_ratio': float((values > 0).mean()) if n else np.nan, 'days': n}


def calibrate(hypotheses, discovery):
    frozen = []
    for hypothesis in hypotheses:
        row = dict(hypothesis)
        values = discovery[row['variable']].dropna()
        if len(values) < 60:
            raise ValueError(f"Insufficient Discovery observations: {row['variable']}")
        row['threshold_value'] = float(values.quantile(row['threshold'] / 100))
        row['discovery_days'] = len(values)
        frozen.append(row)
    return frozen


def evaluate(hypotheses, features, ic):
    baseline = statistics(ic)
    rows = []
    for hypothesis in hypotheses:
        h = hypothesis
        state = features[h['variable']].reindex(ic.index)
        available = state.notna()
        condition = OPERATORS[h['operator']](state, h['threshold_value'])
        counter = statistics(ic[available & condition])
        normal = statistics(ic[available & ~condition])
        row = {k: h[k] for k in ['name', 'variable', 'operator', 'economic_reason', 'expected_effect', 'threshold_value']}
        row['threshold_percentile'] = h['threshold']
        for prefix, stats in [('baseline', baseline), ('counterexample', counter), ('normal', normal)]:
            row.update({f'{prefix}_{key}': value for key, value in stats.items()})
        row['delta_ic'] = counter['ic'] - normal['ic']
        row['missing_condition_days'] = int((~available).sum())
        rows.append(row)
    return pd.DataFrame(rows)
