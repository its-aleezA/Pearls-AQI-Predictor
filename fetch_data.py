import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OWM_KEY    = os.getenv("OWM_KEY")
CITY       = "Rawalpindi"
LAT, LON   = 33.5973, 73.0479
HISTORY_CSV = "aqi_history.csv"


def fetch_waqi() -> dict | None:
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={WAQI_TOKEN}"
    try:
        resp = requests.get(url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        print(f"❌ WAQI network error: {e}")
        return None

    if resp.get("status") != "ok":
        print(f"❌ WAQI API error: {resp.get('data', 'unknown')}")
        return None

    iaqi = resp["data"]["iaqi"]
    return {
        "aqi":      resp["data"]["aqi"],
        "pm25":     iaqi.get("pm25", {}).get("v"),
        "pm10":     iaqi.get("pm10", {}).get("v"),
        "no2":      iaqi.get("no2",  {}).get("v"),
        "o3":       iaqi.get("o3",   {}).get("v"),
        "so2":      iaqi.get("so2",  {}).get("v"),
        "co":       iaqi.get("co",   {}).get("v"),
    }


def fetch_owm() -> dict | None:
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OWM_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10).json()
    except requests.exceptions.RequestException as e:
        print(f"❌ OWM network error: {e}")
        return None

    if resp.get("cod") != 200:
        print(f"❌ OWM API error: {resp.get('message', 'unknown')}")
        return None

    return {
        "temp":       resp["main"]["temp"],
        "humidity":   resp["main"]["humidity"],
        "pressure":   resp["main"]["pressure"],
        "wind_speed": resp["wind"]["speed"],
    }


def fetch_and_save():
    waqi = fetch_waqi()
    owm  = fetch_owm()

    if waqi is None:
        print("⚠️  Skipping save : WAQI fetch failed.")
        return

    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update(waqi)
    # Merge weather; fall back to None if OWM failed this cycle
    row.update(owm if owm else {
        "temp": None, "humidity": None,
        "pressure": None, "wind_speed": None
    })

    df = pd.DataFrame([row])
    header = not os.path.exists(HISTORY_CSV)
    df.to_csv(HISTORY_CSV, mode="a", header=header, index=False)
    print(f"✅ Fetched at {row['timestamp']} — AQI: {row['aqi']}")


if __name__ == "__main__":
    fetch_and_save()
