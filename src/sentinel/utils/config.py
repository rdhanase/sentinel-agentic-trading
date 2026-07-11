"""Configuration loading for the Sentinel dataset pipeline.

Uses a small dataclass with sensible defaults so the pipeline runs even without a YAML file,
and overlays values from ``config/config.yaml`` when present.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

try:
    import yaml  # optional; defaults are used if unavailable
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class DataCfg:
    tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "JPM"])
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    interval: str = "1d"
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"


@dataclass
class FeatureCfg:
    return_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    vol_windows: List[int] = field(default_factory=lambda: [10, 21, 63])
    rsi_period: int = 14
    macd: List[int] = field(default_factory=lambda: [12, 26, 9])
    atr_period: int = 14
    bollinger: List[float] = field(default_factory=lambda: [20, 2])
    volume_zscore_window: int = 21


@dataclass
class LabelCfg:
    horizon: int = 5
    lookback: int = 252
    quantile: float = 0.80


@dataclass
class SplitCfg:
    train_end: str = "2020-12-31"
    valid_end: str = "2022-12-31"
    embargo_days: int = 5


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    label: LabelCfg = field(default_factory=LabelCfg)
    split: SplitCfg = field(default_factory=SplitCfg)

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = "config/config.yaml") -> Config:
    """Load configuration, overlaying YAML values on top of the defaults when available."""
    cfg = Config()
    if path is None or yaml is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    raw = yaml.safe_load(p.read_text()) or {}
    if "data" in raw:
        cfg.data = DataCfg(**{**asdict(cfg.data), **raw["data"]})
    if "features" in raw:
        cfg.features = FeatureCfg(**{**asdict(cfg.features), **raw["features"]})
    if "label" in raw:
        cfg.label = LabelCfg(**{**asdict(cfg.label), **raw["label"]})
    if "split" in raw:
        cfg.split = SplitCfg(**{**asdict(cfg.split), **raw["split"]})
    return cfg
