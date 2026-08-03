# Sentinel — A Volatility-Aware Agentic Stock-Trading Assistant

Sentinel is an agentic stock-trading assistant that combines financial-news sentiment with
market-derived risk signals to make transparent, capital-protective decisions. Its defining feature is a
learned **volatility circuit-breaker**: a supervised neural network that predicts whether the next market
period will enter a high-volatility regime and, when it does, scales down or halts trading. The pieces are
orchestrated with [LangGraph](https://langchain-ai.github.io/langgraph/), and the system runs in
**paper-trading / simulation mode only**.

> **Disclaimer:** This is an educational research project. It is **not financial advice**, uses no real
> capital, and makes no claim about real-market profitability.

## Hypothesis

The learned volatility circuit-breaker reduces maximum drawdown relative to an otherwise identical
strategy that trades without it, while news sentiment adds directional value during normal market regimes.

## Headline result

Over an out-of-sample simulation (2016–2024, equal-weight long book of 16 tickers), switching the
volatility gate on cut the worst drawdown from **-31% to -22%** and raised the Sharpe ratio from
**1.11 to 1.23**, while barely changing average exposure. The gate acts on only about 4% of trading days,
the riskiest ones. The from-scratch neural network reached a walk-forward PR-AUC of 0.44 against a base
rate of 0.21, edging out a tuned XGBoost baseline, and the fine-tuned FinBERT sentiment model reached 0.91
accuracy and 0.90 macro-F1 on held-out financial sentences.

## System overview

| Component | What it is |
|---|---|
| Sentiment model | Fine-tuned **FinBERT** on the Financial PhraseBank (HuggingFace Transformers) |
| Volatility circuit-breaker | From-scratch **PyTorch MLP** (focal loss) with an **XGBoost** baseline |
| Orchestration | **LangGraph** graph: market → sentiment → risk gate → portfolio → execution → rationale |
| Execution | Offline historical **paper-trading simulator** (deterministic, no real orders) |

The circuit-breaker sits in the graph as a conditional edge: when a high-volatility regime is predicted,
the flow routes to a halt/scale branch instead of the normal trading path.

## Repository contents

Code is organized as a small Python package (`src/sentinel/`) with thin scripts and notebooks that call
into it. The required project components map to the files below.

### Data cleaning
| File | Description |
|---|---|
| `src/sentinel/data/acquire.py` | Downloads daily OHLCV from Yahoo Finance (yfinance) with a Stooq fallback. |
| `src/sentinel/data/clean.py` | Fixes duplicate dates, bad/zero prices, inverted bars, and missing values; returns a per-ticker issue report. |
| `scripts/build_dataset.py` | End-to-end dataset build: acquire → clean → features → labels → saved parquet. |

### Feature engineering and labeling
| File | Description |
|---|---|
| `src/sentinel/features/technical.py` | Stationary technical indicators (returns, realized volatility, RSI, MACD, ATR, Bollinger width, volume z-score). |
| `src/sentinel/features/labeling.py` | Forward high-volatility label vs a trailing percentile threshold, plus leakage-safe time splits with an embargo. |
| `config/config.yaml` | Tickers, date range, feature windows, and labeling parameters. |
| `src/sentinel/utils/config.py` | Loads `config.yaml` into typed defaults used across the pipeline. |
| `docs/data_dictionary.md` | Plain-language description of every column in the processed dataset. |

### Exploratory data analysis
| File | Description |
|---|---|
| `src/sentinel/eda/plots.py` | Class balance, correlation matrix, feature-by-regime distributions, and an ADF stationarity report. |
| `scripts/run_eda.py` | Runs the EDA and writes figures to `reports/figures/`. |

### Model design, building, and training
| File | Description |
|---|---|
| `src/sentinel/models/sentiment/data.py` | Loads the Financial PhraseBank and a Twitter out-of-distribution set. |
| `src/sentinel/models/sentiment/finetune.py` | Fine-tunes FinBERT for three-way sentiment (HuggingFace Trainer). |
| `src/sentinel/models/sentiment/infer.py` | `SentimentScorer` and the directional sentiment score used downstream. |
| `src/sentinel/models/volatility/mlp.py` | The from-scratch feed-forward classifier (the required deep-learning model). |
| `src/sentinel/models/volatility/losses.py` | Focal loss, for the rare high-volatility class. |
| `src/sentinel/models/volatility/xgb.py` | Gradient-boosted-tree baseline. |
| `src/sentinel/models/volatility/data.py` | Chronological splits with train-only feature scaling. |

### Model optimization and evaluation
| File | Description |
|---|---|
| `src/sentinel/models/volatility/walk_forward.py` | Walk-forward retraining, the evaluation that suits non-stationary market data. |
| `src/sentinel/models/volatility/evaluate.py` | Imbalanced-aware metrics (PR-AUC) and precision-targeted / max-F1 threshold selection. |
| `scripts/train_volatility.py` | Trains the MLP and XGBoost on a single split and compares them. |
| `scripts/walk_forward_volatility.py` | Walk-forward comparison of the MLP and XGBoost, with per-year breakdown. |

### Agentic pipeline and analysis
| File | Description |
|---|---|
| `src/sentinel/agents/state.py` | The shared `TradingState` passed between nodes. |
| `src/sentinel/agents/nodes.py` | The decision nodes (market, sentiment, risk gate, portfolio, execution, halt, rationale). |
| `src/sentinel/agents/graph.py` | Assembles the nodes into the gated LangGraph flow; `run_day` and `run_backtest`. |
| `src/sentinel/sim/backtest.py` | Offline paper-trading simulator and portfolio metrics. |
| `scripts/save_volatility_oos.py` | Saves the walk-forward out-of-sample predictions the agent and simulator consume. |
| `scripts/run_ablation.py` | The gate-on vs gate-off ablation and its figures. |

### Notebooks and tests
| File | Description |
|---|---|
| `notebooks/main_pipeline.ipynb` | Runs the whole project end to end (data, EDA, volatility model, agent, ablation). |
| `notebooks/sentiment_finbert.ipynb` | Fine-tunes and evaluates FinBERT (executed on a GPU; outputs included). |
| `tests/test_features.py`, `tests/test_labeling.py` | Tests for the technical indicators and the volatility labeling/splits. |
| `tests/test_sentiment.py`, `tests/test_volatility.py` | Tests for the sentiment scoring and the volatility model/metrics. |
| `tests/test_backtest.py`, `tests/test_agents.py` | Tests for the paper-trading simulator and the LangGraph agent. |
| `.github/workflows/ci.yml` | Continuous integration: runs the test suite on push. |

### Supporting files
| File | Description |
|---|---|
| `requirements.txt` | Python dependencies for the whole project. |
| `.gitignore` | Excludes data, model checkpoints, and generated artifacts from version control. |
| `reports/notebook_pdfs/` | PDF exports of the two notebooks. |
| `reports/adf_report.csv` | Saved stationarity-test results from the EDA. |
| `reports/figures/` | Generated figures (git-ignored; produced by the EDA and ablation scripts). |

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the dataset (download -> clean -> features -> labels)
python scripts/build_dataset.py --config config/config.yaml

# 3. Exploratory analysis
python scripts/run_eda.py

# 4. Volatility model: walk-forward comparison (MLP vs XGBoost)
python scripts/walk_forward_volatility.py

# 5. Save out-of-sample predictions, then run the gate-on/off ablation
python scripts/save_volatility_oos.py
python scripts/run_ablation.py

# Tests
pytest -q
```

The FinBERT fine-tuning runs on a GPU and is intended for Colab; see
`notebooks/sentiment_finbert.ipynb`.

## Notes

- Large artifacts (raw/processed data, model checkpoints) are git-ignored; the pipeline regenerates them.
- In the historical backtest the sentiment signal is held neutral because there is no per-day news feed
  aligned to the price history; the live sentiment path is exercised in single-day demonstrations. Wiring
  a historical news feed into the backtest is the main planned extension.

## Use of AI Tools

Generative AI tools (Anthropic Claude, via the Claude Code assistant) were used to help scaffold,
comment, and refactor parts of this codebase and to draft project documentation. All AI-assisted output
was reviewed, executed, tested, and revised by the author, who takes full responsibility for the final
code and results. No data or experimental results were fabricated; every reported metric is produced by
the code in this repository.

## Author

Ramesh Dhanasekaran, M.S. Applied Artificial Intelligence, University of San Diego.
