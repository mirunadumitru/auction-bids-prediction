import numpy as np
import pandas as pd
from pathlib import Path

def main(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["electronics", "home", "fashion", "toys", "motors"]

    start_times = pd.date_range("2025-01-01", periods=n, freq="6H")
    duration_hours = rng.choice([24, 48, 72, 96, 120, 168], size=n)

    df = pd.DataFrame({
        "item_id": np.arange(1_000_000, 1_000_000 + n),
        "category": rng.choice(categories, size=n),
        "start_price": np.round(rng.uniform(5, 500, size=n), 2),
        "start_time": start_times,
        "duration_hours": duration_hours,
    })
    df["end_time"] = df["start_time"] + pd.to_timedelta(df["duration_hours"], unit="h")

    cat_factor = df["category"].map({c: i+1 for i, c in enumerate(categories)}).astype(float)
    time_factor = (df["start_time"].dt.hour/24.0) + (df["start_time"].dt.dayofweek/7.0)
    noise = rng.normal(0, 1, size=n)

    df["num_bids"] = np.maximum(0, (
        9
        + 0.02*(500 - df["start_price"])
        + 1.4*cat_factor
        + 2.2*np.sin(2*np.pi*time_factor)
        + noise
    )).round().astype(int)

    df["end_price"] = np.maximum(
        df["start_price"],
        np.round(df["start_price"] * (1 + 0.01*df["num_bids"] + 0.08*rng.random(n)), 2)
    )

    df["views"] = (60 + 9*df["num_bids"] + rng.normal(0, 12, n)).astype(int).clip(lower=0)
    df["watchlist_count"] = (6 + 0.35*df["num_bids"] + rng.normal(0, 2, n)).astype(int).clip(lower=0)

    raw_path = Path("data/raw/raw.csv")
    sample_path = Path("data/sample.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    df.sample(50, random_state=seed).to_csv(sample_path, index=False)

    print(f"Wrote {raw_path} with {len(df)} rows")
    print(f"Wrote {sample_path} with 50 rows")

if __name__ == "__main__":
    main()
