# feature_engineering.py
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns:
        return df

    # cyclic encoding — hour 23 and hour 0 should be close, not far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_late_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df = df.drop(columns=["hour"])
    return df


def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Amount_scaled" not in df.columns:
        return df

    # interaction between scaled amount and key V features known to correlate with fraud
    # V14 and V17 have the strongest separation between fraud/legit in the real dataset
    for v in ["V14", "V17", "V10", "V12"]:
        if v in df.columns:
            df[f"amt_x_{v}"] = df["Amount_scaled"] * df[v]

    return df


def _pca_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-products of top fraud-discriminating V features.
    These pairs are identified from published EDA on the real ULB dataset —
    they have the largest KS-statistic between fraud and legit distributions.
    """
    top_pairs = [
        ("V14", "V10"), ("V14", "V12"), ("V17", "V10"),
        ("V4",  "V11"), ("V12", "V17"),
    ]
    for a, b in top_pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]

    return df


def _statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return df

    # row-level stats across all V features — fraud rows tend to be outliers
    df["v_mean"]  = df[v_cols].mean(axis=1)
    df["v_std"]   = df[v_cols].std(axis=1)
    df["v_max"]   = df[v_cols].max(axis=1)
    df["v_min"]   = df[v_cols].min(axis=1)
    df["v_range"] = df["v_max"] - df["v_min"]

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    n_start = df.shape[1]

    df = _time_features(df)
    df = _amount_features(df)
    df = _pca_interactions(df)
    df = _statistical_features(df)

    log.info("features: %d → %d columns", n_start, df.shape[1])
    return df


def run(in_path: str, out_path: str):
    log.info("loading %s", in_path)
    df = pd.read_parquet(in_path)

    df = engineer(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="in_path",  default="data/processed/train_clean.parquet")
    parser.add_argument("--out", dest="out_path",  default="data/features/train_features.parquet")
    args = parser.parse_args()
    run(args.in_path, args.out_path)