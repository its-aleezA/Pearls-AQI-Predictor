import os
import joblib
import hopsworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

os.makedirs("predictions", exist_ok=True)

# 1. Login and load model from Hopsworks Model Registry
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))

mr = project.get_model_registry()
# Get the latest version number first, then download
all_versions = mr.get_models("aqi_prediction_model")
latest_version = max(m.version for m in all_versions)
print(f"   Latest model version: {latest_version}")
hw_model  = mr.get_model("aqi_prediction_model", version=latest_version)
model_dir = hw_model.download()

# Load model : try registry download first, fall back to local models/
model_path  = os.path.join(model_dir, "aqi_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

if not os.path.exists(scaler_path):
    print("⚠️  Scaler not in registry download : loading from local models/")
    scaler_path = "models/scaler.pkl"
if not os.path.exists(model_path):
    print("⚠️  Model not in registry download : loading from local models/")
    model_path = "models/aqi_model.pkl"

model  = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("✅ Model and scaler loaded")

# 2. Read features from Feature Store
print("📥 Reading features from Feature Store...")
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_predictions", version=2)
try:
    df = fg.read()
    print(f"   {len(df)} rows loaded from Feature Store")
except Exception as e:
    if "hudi" in str(e).lower() or "no data" in str(e).lower():
        print("⚠️  Hopsworks still materializing : falling back to local CSV...")
        df = pd.read_csv("aqi_history.csv")
        print(f"   {len(df)} rows loaded from aqi_history.csv")
    else:
        raise
df = df.sort_values("timestamp").reset_index(drop=True)

# Gap detection: warn if the most recent record is stale (e.g. due to backfill still running or pipeline issues)
df["timestamp"] = pd.to_datetime(df["timestamp"])
most_recent = df["timestamp"].iloc[-1]
hours_since_last = (datetime.now() - most_recent).total_seconds() / 3600
if hours_since_last > 2:
    print(f"⚠️  WARNING: Most recent feature record is {hours_since_last:.1f}h old "
          f"(last: {most_recent}). Lag features may be skewed : check the feature pipeline.")

# Forward-fill any internal gaps caused by API outages before inference
df = df.ffill()
# Fill any remaining NaNs
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)
df = df.dropna(subset=["aqi"])
print(f"   {len(df)} usable rows after gap-filling")

DROP_COLS    = ["aqi", "timestamp", "pm25"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X      = scaler.transform(df[FEATURE_COLS].values)
y_true = df["aqi"].values

# 3. Historical validation predictions
print("🔮 Running historical batch inference...")
y_pred = model.predict(X)

validation_df = pd.DataFrame({
    "Timestamp":     df["timestamp"].astype(str),
    "Actual_AQI":    y_true,
    "Predicted_AQI": y_pred.round(1),
})
validation_df.to_csv("predictions/aqi_batch_predictions.csv", index=False)
print("💾 Validation predictions → predictions/aqi_batch_predictions.csv")

# 4. 72-hour forward forecast
print("🔭 Generating 72-hour forward forecast...")

last_ts   = df["timestamp"].iloc[-1]
latest_row = df.iloc[-1].to_dict()   # carries last known weather (temp, humidity, etc.)

# Seed the sliding window with the last 24 actual AQI values so rolling stats and lag features are accurate from the very first forecast step
window_history = [
    {"timestamp": row["timestamp"], "aqi": row["aqi"]}
    for _, row in df.tail(24).iterrows()
]

forecast_rows = []
for h in range(1, 73):
    future_ts = last_ts + timedelta(hours=h)

    # Rolling window stats computed from the live sliding buffer
    trailing_6h  = [r["aqi"] for r in window_history[-6:]]
    trailing_24h = [r["aqi"] for r in window_history[-24:]]

    # Build a complete feature row that exactly matches FEATURE_COLS
    row_dict = {
        # Time-based
        "hour":                 future_ts.hour,
        "day_of_week":          future_ts.dayofweek,
        "month":                future_ts.month,
        "is_weekend":           int(future_ts.dayofweek >= 5),
        # Weather : carry forward last known values (no future weather data available)
        "temp":                 latest_row.get("temp"),
        "humidity":             latest_row.get("humidity"),
        "pressure":             latest_row.get("pressure"),
        "wind_speed":           latest_row.get("wind_speed"),
        # Pollutants : carry forward last known
        "pm10":                 latest_row.get("pm10"),
        "no2":                  latest_row.get("no2"),
        "o3":                   latest_row.get("o3"),
        "so2":                  latest_row.get("so2"),
        "co":                   latest_row.get("co"),
        # Change rate
        "aqi_diff":             window_history[-1]["aqi"] - window_history[-2]["aqi"],
        # Lag features from the sliding buffer
        "aqi_lag_1":            window_history[-1]["aqi"],
        "aqi_lag_2":            window_history[-2]["aqi"],
        "aqi_lag_3":            window_history[-3]["aqi"],
        "aqi_lag_24":           window_history[-24]["aqi"] if len(window_history) >= 24
                                else window_history[0]["aqi"],
        # Rolling statistics : recalculated each step from live buffer
        "aqi_rolling_6h_mean":  np.mean(trailing_6h),
        "aqi_rolling_24h_mean": np.mean(trailing_24h),
        "aqi_rolling_6h_std":   np.std(trailing_6h)  if len(trailing_6h)  > 1 else 0.0,
        "aqi_rolling_24h_std":  np.std(trailing_24h) if len(trailing_24h) > 1 else 0.0,
    }

    # Only keep columns the model was trained on, in the right order
    input_df    = pd.DataFrame([row_dict])[FEATURE_COLS]
    feat_scaled = scaler.transform(input_df.values)  # .values silences feature-name warning
    pred_aqi    = max(0.0, round(float(model.predict(feat_scaled)[0]), 1))

    forecast_rows.append({
        "Timestamp":    future_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Forecast_AQI": pred_aqi,
        "Hours_Ahead":  h,
    })

    # Append prediction back into the buffer so the next iteration's lags are correct
    window_history.append({"timestamp": future_ts, "aqi": pred_aqi})

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv("predictions/aqi_72h_forecast.csv", index=False)
print("💾 72-hour forecast → predictions/aqi_72h_forecast.csv")

# 5. Monitoring plot (last 50 actuals + forecast)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: validation
axes[0].plot(validation_df["Actual_AQI"].tail(50).values,
             label="Actual", color="steelblue", marker="o", ms=3)
axes[0].plot(validation_df["Predicted_AQI"].tail(50).values,
             label="Predicted", color="darkorange", linestyle="--", marker="x", ms=3)
axes[0].set_title("Historical: Actual vs Predicted (last 50)")
axes[0].set_xlabel("Data points")
axes[0].set_ylabel("AQI")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: forecast
axes[1].plot(forecast_df["Forecast_AQI"].values,
             color="seagreen", marker=".", ms=4)
axes[1].axhline(100, color="orange", linestyle=":", label="Unhealthy threshold (100)")
axes[1].axhline(150, color="red",    linestyle=":", label="Very Unhealthy (150)")
axes[1].set_title("72-Hour AQI Forecast")
axes[1].set_xlabel("Hours ahead")
axes[1].set_ylabel("AQI")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("predictions/aqi_inference_plot.png", dpi=120)
plt.close()
print("📊 Monitoring plot → predictions/aqi_inference_plot.png")
