# preprocess.py
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load(path: str) -> pd.DataFrame:
    log.info("loading %s", path)
    df = pd.read_csv(path)
    log.info("raw shape: %s", df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # rename target to match pipeline convention
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "isFraud"})

    df["isFraud"] = df["isFraud"].astype(int)

    # no nulls in this dataset but guard anyway
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        log.warning("nulls found in: %s — median filling", null_cols)
        df[null_cols] = df[null_cols].fillna(df[null_cols].median())

    # Amount is raw — scale it so it's on par with V1-V28 (already PCA-scaled)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df = df.drop(columns=["Amount"])

    # Time: convert seconds to hour-of-day (cyclic) — raw seconds isn't useful
    df["hour"] = (df["Time"] % 86400 / 3600).astype(int)
    df = df.drop(columns=["Time"])

    log.info("clean shape: %s  nulls: %d", df.shape, df.isnull().sum().sum())
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    return X, y


def run(in_path: str, out_path: str):
    df = load(in_path)
    df = clean(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="in_path",  default="data/raw/creditcard.csv")
    parser.add_argument("--out", dest="out_path",  default="data/processed/train_clean.parquet")
    args = parser.parse_args()
    run(args.in_path, args.out_path)