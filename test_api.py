"""
test_api.py
-----------
Quick sanity check for both API connections.
Run once locally to confirm your keys work before the pipeline runs.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OWM_KEY    = os.getenv("OWM_KEY")
CITY       = "Rawalpindi"
LAT, LON   = 33.5973, 73.0479


def test_connections():
    print(f"Testing APIs for {CITY}\n")

    # 1. WAQI
    waqi_url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={WAQI_TOKEN}"
    try:
        r = requests.get(waqi_url, timeout=10).json()
        if r.get("status") == "ok":
            d = r["data"]
            print(f"✅ WAQI : AQI: {d['aqi']}, "
                  f"PM2.5: {d['iaqi'].get('pm25',{}).get('v','n/a')}, "
                  f"Station: {d['city']['name']}")
        else:
            print(f"❌ WAQI failed: {r.get('data')}")
    except Exception as e:
        print(f"❌ WAQI exception: {e}")

    # 2. OpenWeatherMap
    owm_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OWM_KEY}&units=metric"
    )
    try:
        r = requests.get(owm_url, timeout=10).json()
        if r.get("cod") == 200:
            m = r["main"]
            print(f"✅ OpenWeather : Temp: {m['temp']}°C, "
                  f"Humidity: {m['humidity']}%, "
                  f"Pressure: {m['pressure']} hPa, "
                  f"Wind: {r['wind']['speed']} m/s")
        else:
            print(f"❌ OWM failed: {r.get('message')}")
    except Exception as e:
        print(f"❌ OWM exception: {e}")

    # 3. OpenAQ
    openaq_url = "https://api.openaq.gov/v3/measurements"
    try:
        r = requests.get(openaq_url,
                         params={"location": CITY, "parameter": "pm25", "limit": 1},
                         timeout=10).json()
        count = len(r.get("results", []))
        print(f"✅ OpenAQ : returned {count} result(s) for {CITY}")
    except Exception as e:
        print(f"❌ OpenAQ exception: {e}")

    # 4. Open-Meteo
    meteo_url = "https://api.open-meteo.com/v1/forecast"
    try:
        r = requests.get(meteo_url,
                         params={"latitude": LAT, "longitude": LON,
                                 "current_weather": True},
                         timeout=10).json()
        cw = r.get("current_weather", {})
        print(f"✅ Open-Meteo : Temp: {cw.get('temperature')}°C, "
              f"Wind: {cw.get('windspeed')} km/h")
    except Exception as e:
        print(f"❌ Open-Meteo exception: {e}")


if __name__ == "__main__":
    test_connections()
