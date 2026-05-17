import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N     = 284807
FRAUD = 492
LEGIT = N - FRAUD

# V feature means for fraud class — from published EDA on real ULB dataset
# legit class centers near 0 (result of PCA transformation)
FRAUD_MEANS = {
    'V1': -4.77, 'V2': 3.95,  'V3': -7.03, 'V4': 4.36,  'V5': -3.15,
    'V6': -1.40, 'V7': -5.57, 'V8': 0.57,  'V9': -2.58, 'V10': -4.58,
    'V11': 4.24, 'V12': -6.80,'V13': 0.07, 'V14': -9.11,'V15': -0.07,
    'V16': -4.57,'V17': -8.13,'V18': -2.56,'V19': 0.36, 'V20': 0.33,
    'V21': 0.76, 'V22': 0.07, 'V23': -0.27,'V24': 0.02, 'V25': 0.06,
    'V26': 0.04, 'V27': 0.47, 'V28': 0.23,
}

print(f"generating {N:,} transactions ({FRAUD/N*100:.3f}% fraud)...")

rows = []

for _ in range(LEGIT):
    row = {f'V{i}': np.random.normal(0.0, 1.5) for i in range(1, 29)}
    row['Time']   = np.random.uniform(0, 172792)
    row['Amount'] = np.random.lognormal(3.0, 1.5)
    row['Class']  = 0
    rows.append(row)

for _ in range(FRAUD):
    row = {f'V{i}': np.random.normal(FRAUD_MEANS[f'V{i}'], 1.2) for i in range(1, 29)}
    row['Time']   = np.random.uniform(0, 172792)
    row['Amount'] = np.random.lognormal(2.0, 1.8)
    row['Class']  = 1
    rows.append(row)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df['Time']   = df['Time'].round(0)
df['Amount'] = df['Amount'].round(2)

out = Path("data/raw/creditcard.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"\n✅  saved → {out}")
print(f"   shape      : {df.shape}")
print(f"   fraud      : {df['Class'].sum()} / {len(df):,}  ({df['Class'].mean()*100:.3f}%)")
print(f"   avg amount : ${df['Amount'].mean():.2f}")
print(f"\nnext steps:")
print(f"   python src/data/preprocess.py --in data/raw/creditcard.csv --out data/processed/train_clean.parquet")
print(f"   python src/data/feature_engineering.py --in data/processed/train_clean.parquet --out data/features/train_features.parquet")