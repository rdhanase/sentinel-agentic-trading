# Sentinel — A Volatility-Aware Agentic Stock-Trading Assistant

Sentinel is a multi-agent stock-trading assistant that combines financial-news sentiment with
market-derived risk signals to make transparent, capital-protective decisions. Its defining feature is a
learned **volatility circuit-breaker**: a supervised model that predicts whether the next market period
will enter a high-volatility regime and, when it does, scales down or halts trading. The system is
orchestrated with [LangGraph](https://langchain-ai.github.io/langgraph/) and runs in **paper-trading /
simulation mode only**.

> **Disclaimer:** This is an educational research project. It is **not financial advice**, uses no real
> capital, and makes no claim about real-market profitability.

## Hypothesis

The learned volatility circuit-breaker meaningfully reduces maximum drawdown relative to an otherwise
identical strategy that trades without it, while news-sentiment signals add incremental directional value
during normal market regimes.

## System at a glance

| Component | What it is | Status |
|---|---|---|
| Sentiment model | Fine-tuned **FinBERT** (HuggingFace) on Financial PhraseBank + FiQA | Planned (M4) |
| Volatility circuit-breaker | From-scratch neural classifier (MLP → CNN-LSTM), XGBoost baseline | Planned (M5) |
| Orchestration | LangGraph pipeline: market → sentiment → risk gate → portfolio → execution → logging | Planned (M6) |
| Execution | Offline historical paper-simulator (Alpaca paper API is a stretch goal) | Planned (M6) |
| **Data pipeline** | **Acquire → clean → engineer features → label volatility regimes** | **In progress (M3)** |

## Repository structure

```
sentinel/
├── config/               # Project configuration (tickers, date ranges, label params)
├── data/
│   ├── raw/              # Immutable downloaded data (git-ignored)
│   ├── interim/          # Cleaned intermediate data
│   ├── processed/        # Model-ready feature/label tables
│   └── external/         # Third-party datasets (e.g., Financial PhraseBank)
├── notebooks/            # EDA and reporting notebooks
├── scripts/              # Pipeline entry points (build_dataset.py, ...)
├── src/sentinel/
│   ├── data/             # acquire.py, clean.py
│   ├── features/         # technical.py (indicators), labeling.py (volatility target)
│   ├── eda/              # plotting / EDA helpers
│   ├── models/           # sentiment/ and volatility/ (M4–M5)
│   ├── agents/           # LangGraph nodes (M6)
│   ├── sim/              # paper-trading simulator (M6)
│   └── utils/            # config loader, helpers
├── tests/                # unit tests (features, labeling)
├── reports/figures/      # generated figures
└── docs/                 # data dictionary, interview prep, notes
```

## Data

- **Market data:** daily OHLCV for a basket of liquid S&P 500 names via
  [`yfinance`](https://pypi.org/project/yfinance/) (Yahoo Finance), with [Stooq](https://stooq.com) as a
  fallback, plus a public Kaggle 1-minute SPY dataset for intraday experiments.
- **News text (M4):** [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
  and FiQA for fine-tuning the sentiment model.

Raw prices are non-stationary, so the model consumes engineered, stationary features (log returns, RSI,
MACD, ATR, Bollinger Band width, rolling realized volatility, normalized volume). The prediction target is
a binary **high-volatility regime** label: 1 when forward realized volatility exceeds a trailing
percentile threshold, 0 otherwise, built with purge/embargo to avoid look-ahead bias.

## Quickstart

```bash
# 1. Create an environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the dataset (download → clean → features → labels)
python scripts/build_dataset.py --config config/config.yaml

# 3. Run the tests
pytest -q
```

Processed feature/label tables land in `data/processed/`, and EDA figures in `reports/figures/`.

## Roadmap (AAI-590, Summer 2026 B)

- **M3 (current):** data acquisition, cleaning, feature engineering, volatility labeling, EDA.
- **M4:** fine-tune FinBERT for financial-news sentiment.
- **M5:** train the volatility circuit-breaker (MLP/CNN-LSTM) vs. an XGBoost baseline.
- **M6:** wire the LangGraph agentic pipeline; run the gate-on vs gate-off ablation in the paper-simulator.
- **M7:** finalize report, repository, and recorded presentation.

## Author

Ramesh Dhanasekaran — M.S. Applied Artificial Intelligence, University of San Diego.

## License

Released under the MIT License. See [LICENSE](LICENSE).
