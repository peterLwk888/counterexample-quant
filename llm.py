"""Minimal OpenAI-compatible Chat Completions client, using only stdlib HTTP."""
import json
import os
import urllib.error
import urllib.request
from features import VARIABLE_DEFINITIONS
from hypothesis import HypothesisBatch, parse_hypotheses
import config


def generate_hypotheses():
    model = os.environ.get('LLM_MODEL')
    if not model:
        raise ValueError('Set LLM_MODEL or supply --hypotheses manual_hypotheses.json.')
    base = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    key = os.environ.get('LLM_API_KEY')
    prompt = f'''Known hypothesis: Stocks with stronger past 20-session returns tend to
have higher returns over the next 5 sessions. Propose exactly {config.HYPOTHESIS_COUNT}
economically meaningful market regimes that could weaken or reverse this relation.
Use ONLY one of these program-defined variables per condition:
{json.dumps(VARIABLE_DEFINITIONS)}
Do not refer to historical events, COVID, financial crises, wars, elections,
specific dates, company names or company news. Use generic economic mechanisms.
No external data or new variables. Operators: >, <, >=, <=.
threshold_type must be percentile, threshold must be a number strictly between 0 and 100,
interpreted using Discovery data only. expected_effect: momentum_rank_ic_decreases.
Return ONLY strict JSON matching this schema, without markdown:
{json.dumps(HypothesisBatch.model_json_schema())}'''
    payload = {'model': model, 'messages': [
        {'role': 'system', 'content': 'You generate falsifiable economic hypotheses. Output strict JSON only.'},
        {'role': 'user', 'content': prompt}],
        'response_format': {'type': 'json_object'}}
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(base + '/chat/completions',
                                     data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'LLM HTTP {exc.code}; check endpoint/model/JSON mode support.') from None
    except urllib.error.URLError:
        raise ValueError('LLM connection failed; check LLM_BASE_URL and network access.') from None
    try:
        choice = body['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('LLM response was incomplete or refused.')
        content = choice['message']['content']
    except (KeyError, IndexError, TypeError):
        raise ValueError('Invalid Chat Completions response.') from None
    return parse_hypotheses(content, config.HYPOTHESIS_COUNT)
