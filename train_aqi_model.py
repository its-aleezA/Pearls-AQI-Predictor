import os
import warnings
import hopsworks
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for CI/CD
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()

# 1. Connect and read features from Hopsworks
print("🔗 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

fg = fs.get_feature_group("aqi_predictions", version=2)
try:
    df = fg.read()
    print(f"✅ Loaded {len(df)} rows from Feature Store")
except Exception as e:
    if "No hudi properties" in str(e) or "no data has been written" in str(e).lower():
        print("⚠️  Hopsworks offline store still materializing : falling back to local CSV...")
        csv_path = "aqi_features.csv" if os.path.exists("aqi_features.csv") else "aqi_history.csv"
        csv_path = "aqi_features.csv" if os.path.exists("aqi_features.csv") else "aqi_history.csv"
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows from {csv_path}")
    else:
        raise

# 2. Prepare features and target 
# Sort chronologically : essential for TimeSeriesSplit
df = df.sort_values("timestamp").reset_index(drop=True)

# Only drop rows where core features are missing : pollutant columns (pm10, no2 etc.) may be all-NaN from the WAQI fallback and would wipe the entire dataset if included
core_cols = ["aqi", "temp", "humidity", "hour", "aqi_lag_1", "aqi_rolling_6h_mean"]
df = df.dropna(subset=[c for c in core_cols if c in df.columns])

# Fill all-NaN columns (e.g. pm10/no2/o3 missing from WAQI fallback) with 0 first, # then fill any remaining partial NaNs with column medians
df = df.fillna(0) if df.isnull().all().any() else df
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)   # catch any columns whose median is also NaN
print(f"   {len(df)} rows after cleaning")

DROP_COLS = ["aqi", "timestamp", "pm25"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURE_COLS].values
y = df["aqi"].values

print(f"📐 Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# 3. TimeSeriesSplit cross-validation (no data leakage) 
tscv = TimeSeriesSplit(n_splits=5)

# 4. Define candidate models 
candidates = {
    "Random_Forest":       RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient_Boosting":   GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Ridge_Regression":    Ridge(alpha=1.0),
}

results = {}

print("\n🤖 Evaluating models with TimeSeriesSplit (5 folds)...")
for name, model in candidates.items():
    fold_rmse, fold_mae, fold_r2 = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Ridge benefits from scaling; tree models don't care but it doesn't hurt
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)

        fold_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
        fold_mae.append(mean_absolute_error(y_te, preds))
        fold_r2.append(r2_score(y_te, preds))

    avg_rmse = np.mean(fold_rmse)
    avg_mae  = np.mean(fold_mae)
    avg_r2   = np.mean(fold_r2)
    results[name] = {"RMSE": round(avg_rmse, 3),
                     "MAE":  round(avg_mae,  3),
                     "R2":   round(avg_r2,   4)}
    print(f"   {name:30s}  RMSE={avg_rmse:.2f}  MAE={avg_mae:.2f}  R²={avg_r2:.4f}")


# 5. Add a simple LSTM deep learning model 
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    WINDOW = 24     # use the past 24 hours to predict the next value

    def make_sequences(X_arr, y_arr, window):
        Xs, ys = [], []
        for i in range(window, len(X_arr)):
            Xs.append(X_arr[i - window:i])
            ys.append(y_arr[i])
        return np.array(Xs), np.array(ys)

    scaler_lstm = StandardScaler()
    X_scaled = scaler_lstm.fit_transform(X)
    Xs, ys = make_sequences(X_scaled, y, WINDOW)

    split = int(len(Xs) * 0.8)
    X_tr_l, X_te_l = Xs[:split], Xs[split:]
    y_tr_l, y_te_l = ys[:split], ys[split:]

    lstm_model = Sequential([
        LSTM(64, input_shape=(WINDOW, X.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=5, restore_best_weights=True)
    lstm_model.fit(
        X_tr_l, y_tr_l,
        epochs=50, batch_size=32,
        validation_split=0.1,
        callbacks=[es], verbose=0,
    )

    lstm_preds = lstm_model.predict(X_te_l).flatten()
    lstm_rmse  = np.sqrt(mean_squared_error(y_te_l, lstm_preds))
    lstm_mae   = mean_absolute_error(y_te_l, lstm_preds)
    lstm_r2    = r2_score(y_te_l, lstm_preds)
    results["LSTM"] = {"RMSE": round(lstm_rmse, 3),
                       "MAE":  round(lstm_mae,  3),
                       "R2":   round(lstm_r2,   4)}
    print(f"   {'LSTM':30s}  RMSE={lstm_rmse:.2f}  MAE={lstm_mae:.2f}  R²={lstm_r2:.4f}")

except ImportError:
    print("   ⚠️  TensorFlow not installed : skipping LSTM (pip install tensorflow)")

# 6. Select champion 
best_name = min(results, key=lambda k: results[k]["RMSE"])
best_metrics = results[best_name]
print(f"\n🏆 Champion: {best_name}  →  {best_metrics}")

# Re-train champion on ALL data with a final scaler
final_scaler = StandardScaler()
X_all_s = final_scaler.fit_transform(X)

if best_name == "LSTM":
    champion = lstm_model       # already trained on full window data
else:
    champion = candidates[best_name]
    champion.fit(X_all_s, y)

# 7. SHAP feature importance (tree models only) 
os.makedirs("models", exist_ok=True)
shap_plot_path = "models/shap_summary.png"

if best_name in ("Random_Forest", "Gradient_Boosting"):
    print("📊 Computing SHAP values...")
    explainer   = shap.TreeExplainer(champion)
    shap_values = explainer.shap_values(X_all_s[:500])   # sample for speed

    plt.figure()
    shap.summary_plot(shap_values, X_all_s[:500],
                      feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   SHAP plot saved → {shap_plot_path}")
else:
    # For Ridge / LSTM, use permutation-based importance as fallback
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        champion if best_name != "LSTM" else Ridge().fit(X_all_s, y),
        X_all_s, y, n_repeats=10, random_state=42
    )
    plt.figure(figsize=(8, 5))
    sorted_idx = perm.importances_mean.argsort()[::-1]
    plt.barh([FEATURE_COLS[i] for i in sorted_idx],
             perm.importances_mean[sorted_idx])
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=120, bbox_inches="tight")
    plt.close()

# 8. Save model + scaler locally 
model_path  = "models/aqi_model.pkl"
scaler_path = "models/scaler.pkl"

if best_name == "LSTM":
    champion.save("models/aqi_lstm_model.keras")
    model_path = "models/aqi_lstm_model.keras"
else:
    joblib.dump(champion, model_path)

joblib.dump(final_scaler, scaler_path)
print(f"💾 Model saved → {model_path}")

# 9. Register in Hopsworks Model Registry 
print("📤 Registering champion model in Hopsworks...")
mr = project.get_model_registry()

def sanitize_metric(v):
    import math
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0

hw_model = mr.python.create_model(
    name="aqi_prediction_model",
    metrics={
        "RMSE": sanitize_metric(best_metrics["RMSE"]),
        "MAE":  sanitize_metric(best_metrics["MAE"]),
        "R2":   sanitize_metric(best_metrics["R2"]),
    },
    description=(
        f"Champion: {best_name}. "
        f"All model results: {results}. "
        f"Features: {FEATURE_COLS}"
    ),
)
hw_model.save("models/")   # uploads entire models/ dir (model + scaler + SHAP plot)
print("🚀 Model registered in Hopsworks Model Registry!")

# Print full comparison table
print("\n📋 Full model comparison:")
print(f"{'Model':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 58)
for name, m in results.items():
    marker = " ← champion" if name == best_name else ""
    print(f"{name:<30} {m['RMSE']:>8} {m['MAE']:>8} {m['R2']:>8}{marker}")
