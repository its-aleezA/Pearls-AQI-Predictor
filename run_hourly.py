import time
import os
import subprocess

print("🚀 Hourly Data Fetcher Started... Keep this window open!")

while True:
    print(f"[{time.strftime('%H:%M:%S')}] Fetching new data...")
    # This runs the existing script
    subprocess.run(["python", "fetch_data.py"])
    
    # Run feature engineering to update the 'aqi_features.csv'
    subprocess.run(["python", "feature_engineering.py"])
    
    print("✅ Done. Sleeping for 1 hour...")
    # Wait for 3600 seconds (1 hour)
    time.sleep(3600)