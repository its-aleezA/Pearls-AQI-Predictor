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

fg.insert(df)
print(f"🚀 {len(df)} rows pushed to Hopsworks Feature Store!")
