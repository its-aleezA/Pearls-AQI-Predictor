import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project  = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
mr       = project.get_model_registry()

print("📡 Fetching latest model from registry...")
hw_model = mr.get_model("aqi_prediction_model", version=1)
local_dir = hw_model.download()

print(f"✅ Downloaded to: {local_dir}")
print("   Files:")
for f in os.listdir(local_dir):
    print(f"     {f}")
