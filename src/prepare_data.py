"""
prepare_data.py
===============
Run this ONCE locally before pushing to GitHub.

It creates two lightweight CSV files that the Streamlit app uses on the cloud:
  - train_sample.csv  (~50 K rows, <12 MB  -> safe for GitHub & Streamlit Cloud RAM)
  - test_sample.csv   (full test set kept,   ~6.8 MB)

Usage (from inside the src/ folder):
    python prepare_data.py
"""

from pathlib import Path
import pandas as pd

SRC = Path(__file__).parent

NEEDED_COLS = [
    "town", "flat_type", "hdb_age", "mid", "resale_price",
    "floor_area_sqm",
    "mrt_nearest_distance", "pri_sch_nearest_distance",
    "bus_stop_nearest_distance", "hawker_within_1km", "mall_within_1km",
]

SAMPLE_ROWS = 60_000   # ~12 MB on disk; well under GitHub's 50 MB cap
RANDOM_SEED = 42


def load_raw(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    # keep only columns that exist
    cols = [c for c in NEEDED_COLS if c in df.columns]
    return df[cols]


print("Reading train_clean_working.csv …")
train = load_raw(SRC / "train_clean_working.csv")
print(f"  Full train shape: {train.shape}")

sample = train.sample(n=min(SAMPLE_ROWS, len(train)), random_state=RANDOM_SEED)
out_train = SRC / "train_sample.csv"
sample.to_csv(out_train, index=False)
print(f"  Saved {len(sample):,} rows -> {out_train.name}  "
      f"({out_train.stat().st_size / 1e6:.1f} MB)")

print("Reading test_clean_working.csv …")
test = load_raw(SRC / "test_clean_working.csv")
print(f"  Full test shape: {test.shape}")

out_test = SRC / "test_sample.csv"
test.to_csv(out_test, index=False)
print(f"  Saved {len(test):,} rows -> {out_test.name}  "
      f"({out_test.stat().st_size / 1e6:.1f} MB)")

print("\nDone! Commit train_sample.csv and test_sample.csv to GitHub.")
print("Keep the original *_clean_working.csv files in .gitignore.")
