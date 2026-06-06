import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time-based features 
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # AQI change rate 
    df["aqi_diff"] = df["aqi"].diff().fillna(0)

    # Lag features (temporal memory) 
    # These are the strongest predictors for short-horizon AQI forecasting
    for lag in [1, 2, 3, 24]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # Rolling statistics 
    df["aqi_rolling_6h_mean"]  = df["aqi"].rolling(window=6,  min_periods=1).mean()
    df["aqi_rolling_24h_mean"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    df["aqi_rolling_6h_std"]   = df["aqi"].rolling(window=6,  min_periods=1).std().fillna(0)
    df["aqi_rolling_24h_std"]  = df["aqi"].rolling(window=24, min_periods=1).std().fillna(0)

    return df


if __name__ == "__main__":
    try:
        raw = pd.read_csv("aqi_history.csv")
        processed = engineer_features(raw)
        processed.to_csv("aqi_features.csv", index=False)
        print("✅ Feature engineering complete — saved to aqi_features.csv")
        print(processed[["timestamp", "aqi", "hour", "aqi_diff",
                          "aqi_lag_1", "aqi_rolling_24h_mean"]].tail())
    except FileNotFoundError:
        print("❌ Run fetch_data.py first to create aqi_history.csv")
