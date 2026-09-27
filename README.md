# Counterexample Quant

## Research Question
Can LLMs identify economically meaningful failure regimes of known quantitative factors?

## What this project is NOT
This project is not designed to let an LLM directly predict stock returns.
It is not an automated alpha mining system.
Instead, the LLM acts as a hypothesis / counterexample generator, while all empirical evaluation is deterministic.
No trading, frontend, database, or agent framework.

## Pipeline
Known Alpha → LLM Counterexample Generation → Formalized Condition → Freeze → Out-of-Sample Evaluation

V0 studies only 20-session price momentum and daily cross-sectional Spearman Rank IC against subsequent 5-session returns. SPY supplies market state variables. Discovery: 2015–2021; Test: 2022 onward. Settings live in `config.py`.

## Development
Implemented incrementally: data → trailing features → factor/IC → manual condition evaluation → strict schema → LLM → random conditions → end-to-end verification.

## Quick start
Python 3.11+ is recommended (tested on 3.12).

```bash
cd counterexample-quant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

The default run downloads Yahoo daily data and uses the five **handwritten** examples in `manual_hypotheses.json`; these are plumbing examples, not evidence about LLM performance. No API key is needed. `results/hypotheses.json` is reused once it exists. Manual JSON uses the same schema as the API and must contain exactly five hypotheses by default.

```bash
python run.py --hypotheses my_hypotheses.json
python run.py --csv daily_prices.csv
python -m unittest -v
```

CSV format (one row per date and ticker, including SPY):

```csv
date,ticker,price,volume
2015-01-02,AAPL,24.25,212818400
2015-01-02,SPY,173.17,121465900
```

These two rows illustrate format only. Supply the full configured universe and periods. `price` must use one consistent adjusted-close convention; the loader cannot infer corporate-action adjustments. Dates must be daily session dates. Do not mix adjusted and unadjusted prices. Missing prices remain missing; there is no fill or fallback to synthetic data. The market's observed sessions define the calendar. Missing whole SPY sessions in a source cannot be detected without an exchange calendar; verify source completeness before interpreting results.

## LLM generation
For a first LLM experiment, use a separate results directory from the manual smoke test:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="your-provider-model"
# Set LLM_API_KEY in your shell or secret manager, never in source files.
python run.py --llm --results-dir results/llm_experiment
# Later: reuse frozen hypotheses and thresholds, without another API call.
python run.py --results-dir results/llm_experiment
```

`LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are read only from the environment. A local server can omit the key. The base URL should include `/v1` when the provider uses it; the client appends `/chat/completions`. The provider must support Chat Completions JSON mode. There are no automatic retries or format repairs. JSON mode itself does not guarantee schema compliance, so Pydantic validates the complete response: [OpenAI structured output documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

The LLM receives only the known factor hypothesis, seven variable definitions and the output schema. It receives no prices, dates, stocks, Discovery returns, or Test statistics. Exact variables/operators/effect/percentiles are validated; unknown fields, duplicate keys, duplicate conditions and invalid batches are rejected. Historical event/date references are forbidden in the prompt and common references are rejected by a conservative text check. This is not a semantic guarantee: review economic explanations for company-news references, unsupported variables and plausibility **before inspecting Test results**. Model pretraining can contain historical knowledge; the prompt cannot eliminate that contamination.

## Features and timing
All signals are observed **after the close of day t**. This is a statistical factor evaluation, not an executable same-close trading simulation.

| Variable | Deterministic definition |
| --- | --- |
| `market_return_5` | SPY adjusted close / close 5 sessions ago − 1 |
| `market_return_20` | SPY adjusted close / close 20 sessions ago − 1 |
| `market_volatility_20` | Sample std of last 20 SPY daily returns × √252 |
| `market_drawdown_60` | SPY close / maximum close over last 60 sessions − 1 |
| `market_volume_zscore_20` | (Current SPY volume − trailing 20-session mean) / sample std |
| `cross_sectional_dispersion` | Sample std of individual stock daily returns, excluding SPY |
| `momentum_dispersion` | Sample std of stock momentum, excluding SPY |

Trailing windows include t; z-score uses that same convention. All rolling windows require their full number of observations. Zero volume standard deviation gives a missing z-score. Cross-sectional features and IC require at least `MIN_STOCKS` (default 10) valid names. Rank IC uses only paired valid momentum/forward-return observations and average ranks for ties; constant ranks produce missing IC. Missing condition values are excluded from **both** subsets and counted separately. Thus baseline day count can exceed the two subset counts combined.

`momentum_20 = price_t / price_(t-20) - 1` and `future_return_5 = price_(t+5) / price_t - 1`. The baseline includes every Test signal date with valid IC and a fully observed 5-session label within the configured Test end. Last five sessions have no mature labels and are excluded. Test features may legitimately use preceding Discovery history; labels never enter features or percentile calibration. The current New York calendar day is excluded to avoid incomplete daily bars. Date bounds in config are inclusive; the yfinance exclusive end is converted explicitly. Adjusted close is requested explicitly via `auto_adjust=True`: [yfinance implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/multi.py).

## Freeze and data snooping
1. Validate and save hypotheses to `results/hypotheses.json` before evaluation.
2. Slice price/volume inputs at Discovery end; build trailing features from that prefix only.
3. Calculate each percentile using nonmissing Discovery feature values (`pandas.quantile`, linear interpolation). At least 60 valid observations per variable are required. No optimization against Discovery IC is performed.
4. Draw equally many distinct random conditions from the same variables, `>`/`<`, and percentiles 10/20/30/70/80/90, with a fixed seed. Calibrate on Discovery only.
5. Save both sets of numeric conditions and protocol to `frozen_conditions.json`, plus the exact Discovery price/volume bars to `discovery_data.csv`.
6. Only then compute Test IC, subsets and reports.

Reruns reuse frozen thresholds. Configuration and hypothesis fingerprints are enforced; source-code changes produce a warning so that a bug fix does not silently reset an experiment. When a Discovery snapshot exists, its historical bars replace revised vendor values on rerun; missing or changed snapshot data cause an error. Older experiments created before this snapshot feature retain their original thresholds when minor vendor revisions occur, with an explicit warning that exact reproduction is impossible. Material revisions still cause an error. An open Test end permits new observations on later runs; historical vendor revisions can still change Test results. No test-driven threshold or seed selection is implemented. Separate directories are for planned experiments, not for repeatedly sampling hypotheses until Test looks favorable. Once observed, this historical Test is no longer fresh evidence for future iterations; use genuinely unseen periods for later confirmation.

## Outputs and interpretation
- `hypotheses.json`: original validated hypotheses.
- `frozen_conditions.json`: calibrated conditions, random conditions, provenance and protocol fingerprints.
- `discovery_data.csv`: historical price and volume snapshot for exact Discovery feature reconstruction (new experiments).
- `evaluation.csv` / `random_evaluation.csv`: all hypotheses, including empty or unfavorable subsets.
- `daily_ic.csv`: Test IC audit series.
- `delta_summary.csv`: count, mean, standard deviation, min, quartiles and max of each group's delta IC.

CSV fields include names, reasons, variables/operators, percentile and numeric threshold. Each of baseline/counterexample/normal has mean IC, sample IC std, naive t-statistic, positive IC ratio and number of days. `delta_ic = counterexample_ic - normal_ic`. A negative value describes weaker momentum in that condition; a negative conditional mean describes reversal in this sample. Neither establishes statistical significance or a profitable strategy. Empty subsets have blank/NaN statistics; one observation has no sample standard deviation or t-statistic. A zero standard deviation gives an undefined t-statistic.

The reported t-statistic is `mean / (sample_std / sqrt(n))`, labeled **naive**: 5-session overlapping returns and persistent states induce serial dependence. It is not a valid independence-adjusted significance test. V0 deliberately omits HAC, bootstrap and multiple-testing correction. Five random conditions provide only a small descriptive comparison, not evidence that LLM hypotheses outperform chance. This project does not claim to establish “significant deterioration” with its V0 statistics.

The fixed, present-day large-cap list carries survivorship/selection bias and is not representative of all historical US equities. Adjusted vendor series are not point-in-time data vintages; missing data and revisions can affect results. These constraints limit the research claim even though code avoids future feature/threshold leakage. There are no trading costs, execution rules or portfolio returns.

## Modules and verification
`data.py` loads/aligns daily tables; `features.py` constructs trailing states; `factor.py` computes momentum, forward labels and IC; `evaluator.py` calibrates/evaluates deterministic conditions; `hypothesis.py` validates JSON and samples random conditions; `llm.py` makes one API request; `run.py` freezes and orchestrates.

The offline unittest suite uses explicitly synthetic data and a mocked API. It verifies prefix invariance, Test perturbation isolation, factor formulas, forward-label boundaries, pairwise ranking with ties/missing observations, schema rejection, empty subsets, random reproducibility, the API contract, CSV end-to-end execution and frozen-protocol protection. Synthetic outputs are software checks, not empirical findings.
