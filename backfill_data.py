import os
import time
import requests
import pandas as pd
import hopsworks
from datetime import datetime, timedelta
from dotenv import load_dotenv
from feature_engineering import engineer_features

load_dotenv()

# Config
LAT, LON      = 33.5973, 73.0479
BACKFILL_DAYS = 60
WAQI_TOKEN    = os.getenv("WAQI_TOKEN")


# Fetch AQI history from WAQI
def fetch_waqi_history(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"📡 Fetching WAQI history from {start_date} to {end_date}...")

    start   = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date,   "%Y-%m-%d")
    rows    = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = (
            f"https://api.waqi.info/api/feed/@{LAT};{LON}/history.json"
            f"?token={WAQI_TOKEN}&date={date_str}"
        )
        try:
            resp = requests.get(url, timeout=10).json()
            if resp.get("status") == "ok":
                data    = resp.get("data", {})
                aqi_val = data.get("aqi")
                iaqi    = data.get("iaqi", {})
                if aqi_val and str(aqi_val).lstrip("-").isdigit():
                    rows.append({
                        "timestamp": f"{date_str} 12:00",
                        "aqi":       int(aqi_val),
                        "pm25":      iaqi.get("pm25", {}).get("v"),
                        "pm10":      iaqi.get("pm10", {}).get("v"),
                        "no2":       iaqi.get("no2",  {}).get("v"),
                        "o3":        iaqi.get("o3",   {}).get("v"),
                        "so2":       iaqi.get("so2",  {}).get("v"),
                        "co":        iaqi.get("co",   {}).get("v"),
                    })
        except requests.exceptions.RequestException as e:
            print(f"   warning: error fetching {date_str}: {e}")

        current += timedelta(days=1)
        time.sleep(0.3)

    # Fallback: use forecast.daily embedded in the live feed
    if not rows:
        print("   WAQI history endpoint empty : trying forecast.daily fallback...")
        url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={WAQI_TOKEN}"
        try:
            resp = requests.get(url, timeout=10).json()
            if resp.get("status") == "ok":
                daily = resp["data"]["forecast"]["daily"].get("pm25", [])
                for entry in daily:
                    aqi_est = min(int(float(entry["avg"]) * 100 / 35.4), 500)
                    rows.append({
                        "timestamp": f"{entry['day']} 12:00",
                        "aqi":  aqi_est,
                        "pm25": entry["avg"],
                        "pm10": None, "no2": None,
                        "o3":   None, "so2": None, "co": None,
                    })
        except Exception as e:
            print(f"   fallback also failed: {e}")

    if not rows:
        print("❌ WAQI returned no usable data.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    print(f"   ✅ {len(df)} daily readings retrieved from WAQI")
    return df


# Fetch weather history from Open-Meteo (free, no key)
def fetch_openmeteo_history(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"📡 Fetching Open-Meteo weather from {start_date} to {end_date}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     "temperature_2m,relativehumidity_2m,surface_pressure,windspeed_10m",
        "timezone":   "Asia/Karachi",
    }
    try:
        resp = requests.get(url, params=params, timeout=20).json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Open-Meteo network error: {e}")
        return pd.DataFrame()

    hourly = resp.get("hourly", {})
    if not hourly:
        print("⚠️  No Open-Meteo data returned.")
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp":  hourly["time"],
        "temp":       hourly["temperature_2m"],
        "humidity":   hourly["relativehumidity_2m"],
        "pressure":   hourly["surface_pressure"],
        "wind_speed": hourly["windspeed_10m"],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    print(f"   ✅ {len(df)} hourly weather rows retrieved from Open-Meteo")
    return df


# Main backfill routine
def run_backfill():
    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=BACKFILL_DAYS)
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    df_aqi     = fetch_waqi_history(start_str, end_str)
    df_weather = fetch_openmeteo_history(start_str, end_str)

    if df_aqi.empty:
        print("❌ AQI backfill failed : no data to process.")
        return

    expanded = []
    for _, row in df_aqi.iterrows():
        base_date = pd.to_datetime(row["timestamp"]).date()
        for h in range(24):
            r = row.to_dict()
            r["timestamp"] = f"{base_date} {h:02d}:00"
            expanded.append(r)
    df_aqi_hourly = pd.DataFrame(expanded)

    if not df_weather.empty:
        df = pd.merge(df_aqi_hourly, df_weather, on="timestamp", how="left")
    else:
        df = df_aqi_hourly.copy()
        for col in ["temp", "humidity", "pressure", "wind_speed"]:
            df[col] = None

    # Engineer features
    df_features = engineer_features(df)
    df_features = df_features.dropna(subset=["aqi"])

    # Fix dtypes before Hopsworks insert
    # Float columns: all nullable pollutant + weather columns
    float_cols = ["pm25", "pm10", "no2", "o3", "so2", "co",
                  "temp", "humidity", "pressure", "wind_speed",
                  "aqi_diff", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_24",
                  "aqi_rolling_6h_mean", "aqi_rolling_24h_mean",
                  "aqi_rolling_6h_std",  "aqi_rolling_24h_std"]
    for col in float_cols:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce").astype("float64")

    # Integer columns
    int_cols = ["aqi", "hour", "day_of_week", "month", "is_weekend"]
    for col in int_cols:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce").astype("int64")

    print(f"💾 {len(df_features)} rows engineered : saving locally...")
    df_features.to_csv("aqi_history.csv", index=False)

    # Push to Hopsworks Feature Store
    print("📤 Uploading to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_predictions",
        version=2,
        primary_key=["timestamp"],
        description="Hourly AQI + weather features for Rawalpindi : full schema v2",
    )
    df_features["timestamp"] = df_features["timestamp"].astype(str)
    fg.insert(df_features)
    print(f"🚀 Backfill complete : {len(df_features)} rows pushed to Hopsworks!")


if __name__ == "__main__":
    run_backfill()
