"""Strict JSON boundary: reject invalid batches rather than repair or select."""
import json
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from features import VARIABLE_DEFINITIONS


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1, max_length=120)
    economic_reason: str = Field(min_length=1, max_length=2000)
    variable: Literal['market_return_5', 'market_return_20', 'market_volatility_20',
                      'market_drawdown_60', 'market_volume_zscore_20',
                      'cross_sectional_dispersion', 'momentum_dispersion']
    operator: Literal['>', '<', '>=', '<=']
    threshold_type: Literal['percentile']
    threshold: float = Field(gt=0, lt=100, allow_inf_nan=False)
    expected_effect: Literal['momentum_rank_ic_decreases']

    @field_validator('name', 'economic_reason')
    @classmethod
    def generic_reason(cls, value):
        if not value.strip():
            raise ValueError('Text must not be blank.')
        # Conservative guard, not a semantic proof. Human review is still necessary.
        forbidden = r'covid|coronavirus|pandemic|financial.crisis|\bwar\b|\belection\b|\b(?:19|20)\d{2}\b|\d{1,2}[/-]\d{1,2}|新冠|疫情|金融危机|战争|选举|新闻'
        if re.search(forbidden, value, re.I):
            raise ValueError('Use generic economic mechanisms, not named historical events/dates/news.')
        return value


class HypothesisBatch(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    hypotheses: list[Hypothesis] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_conditions(self):
        names = [h.name for h in self.hypotheses]
        conditions = [(h.variable, h.operator, h.threshold) for h in self.hypotheses]
        if len(set(names)) != len(names) or len(set(conditions)) != len(conditions):
            raise ValueError('Duplicate names or conditions are not permitted.')
        return self


def parse_hypotheses(text, expected_count=None):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    obj = json.loads(text, object_pairs_hook=unique_keys)
    batch = HypothesisBatch.model_validate(obj)
    if expected_count is not None and len(batch.hypotheses) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} hypotheses.')
    return [h.model_dump() for h in batch.hypotheses]


def random_hypotheses(count, seed):
    import random
    from itertools import product
    candidates = list(product(VARIABLE_DEFINITIONS, ['>', '<'], [10, 20, 30, 70, 80, 90]))
    choices = random.Random(seed).sample(candidates, count)
    return [Hypothesis(name=f'random_{i}', economic_reason='Random comparison condition; no economic selection.',
                       variable=variable, operator=op, threshold_type='percentile', threshold=float(percentile),
                       expected_effect='momentum_rank_ic_decreases').model_dump()
            for i, (variable, op, percentile) in enumerate(choices, 1)]
