# Data Dictionary

Processed dataset: `data/processed/features_labeled.parquet` (one row per ticker-day).

## Identifiers and raw fields

| Column | Description |
|---|---|
| `date` | Trading date (tz-naive). |
| `ticker` | Equity/ETF symbol. |
| `close` | Unadjusted closing price. |
| `adj_close` | Split/dividend-adjusted closing price. |
| `volume` | Shares traded. |

## Engineered features (all stationary)

| Column | Description |
|---|---|
| `ret_1`, `ret_5`, `ret_10`, `ret_20` | Log returns over 1/5/10/20 trading days. |
| `rvol_10`, `rvol_21`, `rvol_63` | Rolling realized volatility (std of daily log returns) over 10/21/63 days. |
| `rsi` | Wilder's Relative Strength Index (14), scaled to [0, 1]. |
| `macd_line`, `macd_signal`, `macd_hist` | MACD (12/26/9), each divided by price to be scale-free. |
| `atr` | Average True Range (14) as a fraction of close. |
| `bb_width` | Bollinger Band width (20, 2 std) = (upper − lower) / middle. |
| `vol_z` | Volume z-score over a 21-day trailing window. |
| `hl_range` | Daily high−low range as a fraction of close. |

## Target and helper columns

| Column | Description |
|---|---|
| `fwd_rvol` | Forward realized volatility over the next H=5 days (uses future returns; **label only**). |
| `vol_threshold` | Trailing 80th-percentile realized-vol threshold (past data only, L=252). |
| `label_highvol` | **Target.** 1 if `fwd_rvol` > `vol_threshold` (high-vol regime), else 0. `NA` at series edges. |

## Leakage controls

- `fwd_rvol` looks forward and appears only in the label, never as a feature.
- `vol_threshold` is computed from a trailing window shifted by one day, so no future information sets it.
- Train/valid/test splits are chronological with an embargo (≥ label horizon) at each boundary; see
  `sentinel.features.labeling.time_split`.
