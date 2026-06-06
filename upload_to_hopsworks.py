import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv
 
load_dotenv()
 
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
print(f"🔗 Connected to Hopsworks project: {project.name}")
 
# Load the locally engineered features
try:
    df = pd.read_csv("aqi_features.csv")
except FileNotFoundError:
    print("❌ aqi_features.csv not found. Run feature_engineering.py first.")
    exit(1)
 
df["timestamp"] = (
    pd.to_datetime(df["timestamp"], format="mixed")
    .dt.strftime("%Y-%m-%d %H:%M:%S")
)
 
# Get or create the feature group
fg = fs.get_or_create_feature_group(
    name="aqi_predictions",
    version=2,
    primary_key=["timestamp"],
    description="Hourly AQI + weather features for Rawalpindi",
)
 
# Cast columns to correct types : Hopsworks rejects type mismatches
float_cols = ["pm25", "pm10", "no2", "o3", "so2", "co",
              "temp", "humidity", "pressure", "wind_speed",
              "aqi_diff", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_24",
              "aqi_rolling_6h_mean", "aqi_rolling_24h_mean",
              "aqi_rolling_6h_std", "aqi_rolling_24h_std"]
int_cols = ["aqi", "hour", "day_of_week", "month", "is_weekend"]
 
for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
 
fg.insert(df)
print(f"🚀 {len(df)} rows pushed to Hopsworks Feature Store!")
