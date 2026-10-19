import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def prepare_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["day_number"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_number"] >= 5).astype(int)

    return df


def train_and_predict(df, forecast_days=7):

    df = prepare_features(df)

    results = []
    metrics = []

    features = [
        "day_number",
        "month",
        "day_of_month",
        "is_weekend"
    ]

    for dish in df["dish"].unique():

        dish_df = (
            df[df["dish"] == dish]
            .sort_values("date")
            .copy()
            .reset_index(drop=True)
        )

        X = dish_df[features]
        y = dish_df["sold"]

        split = int(len(dish_df) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        X_test = X.iloc[split:]
        y_test = y.iloc[split:]

        # MODEL 1: RANDOM FOREST

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        )

        model.fit(X_train, y_train)

        rf_predictions = model.predict(X_test)

        rf_mae = mean_absolute_error(
            y_test,
            rf_predictions
        )

        # MODEL 2: SAME-WEEKDAY BASELINE
        # Predict using the same weekday from 7 days earlier.
        baseline_predictions = (
            dish_df["sold"]
            .shift(7)
            .iloc[split:]
        )

        # First test week may not have enough previous data.
        valid = baseline_predictions.notna()

        if valid.any():

            baseline_mae = mean_absolute_error(
                y_test.loc[valid],
                baseline_predictions.loc[valid]
            )

        else:

            baseline_mae = float("nan")

        # Train on all historical data for future forecasts.
        model.fit(X, y)

        last_date = dish_df["date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        future = pd.DataFrame({
            "date": future_dates
        })

        future["day_number"] = future["date"].dt.dayofweek
        future["month"] = future["date"].dt.month
        future["day_of_month"] = future["date"].dt.day
        future["is_weekend"] = (
            future["day_number"] >= 5
        ).astype(int)

        future["predicted_demand"] = (
            model.predict(future[features])
            .round()
            .clip(min=0)
            .astype(int)
        )

        future["dish"] = dish

        results.append(future)

        metrics.append({
            "dish": dish,
            "random_forest_mae": round(rf_mae, 2),
            "weekday_baseline_mae": (
                round(baseline_mae, 2)
                if pd.notna(baseline_mae)
                else None
            ),
            "test_records": len(y_test)
        })

    forecast_df = pd.concat(
        results,
        ignore_index=True
    )

    metrics_df = pd.DataFrame(metrics)

    return forecast_df, metrics_df


if __name__ == "__main__":

    df = pd.read_csv("data/canteen_data.csv")

    forecasts, metrics = train_and_predict(df)

    print("\n7-DAY DEMAND FORECAST")

    print(
        forecasts[
            ["date", "dish", "predicted_demand"]
        ].to_string(index=False)
    )

    print("\nMODEL COMPARISON")

    print(metrics.to_string(index=False))