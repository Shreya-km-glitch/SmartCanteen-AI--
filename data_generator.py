import numpy as np
import pandas as pd
from pathlib import Path


def generate_canteen_data(days=180, seed=42):
    rng = np.random.default_rng(seed)

    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=days,
        freq="D"
    )

    dishes = {
        "Rajma Rice": (150, 35, 60),
        "Chole Bhature": (130, 30, 55),
        "Veg Thali": (180, 45, 80),
        "Masala Dosa": (100, 25, 50),
    }

    records = []

    for i, date in enumerate(dates):
        weekday = date.weekday()
        day_factor = 0.65 if weekday >= 5 else 1.0
        trend = 1 + 0.0008 * i

        for dish, (base, cost, price) in dishes.items():
            expected = base * day_factor * trend

            demand = max(
                0,
                int(rng.normal(expected, 10))
            )

            prepared = max(
                0,
                int(demand + rng.normal(12, 8))
            )

            sold = min(prepared, demand)
            leftovers = prepared - sold

            records.append({
                "date": date,
                "day": date.day_name(),
                "dish": dish,
                "prepared": prepared,
                "sold": sold,
                "leftover_portions": leftovers,
                "leftover_kg": round(leftovers * 0.25, 2),
                "cost_per_portion": cost,
                "price_per_portion": price,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    df = generate_canteen_data()

    output_file = output_dir / "canteen_data.csv"
    df.to_csv(output_file, index=False)

    print("SUCCESS! Dataset generated.")
    print(f"Total records: {len(df)}")
    print(f"Saved to: {output_file.resolve()}")
    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))