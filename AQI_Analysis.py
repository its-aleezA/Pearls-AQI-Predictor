# %% [markdown]
# # Rawalpindi AQI — Exploratory Data Analysis
# End-to-end EDA: trends, correlations, seasonal patterns, and feature insights.

# %% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("aqi_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
df.describe().round(2)

# %% [markdown]
# ## 1. AQI trend over time

# %%
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["timestamp"], df["aqi"], linewidth=0.8, color="steelblue", label="AQI")
ax.axhline(100, color="orange", linestyle="--", linewidth=0.8, label="Unhealthy threshold")
ax.axhline(150, color="red",    linestyle="--", linewidth=0.8, label="Very Unhealthy")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.set_title("AQI Over Time — Rawalpindi")
ax.set_ylabel("AQI")
ax.legend()
plt.tight_layout()
plt.savefig("eda_trend.png", dpi=120)
plt.show()

# %% [markdown]
# ## 2. AQI distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["aqi"], bins=40, color="steelblue", edgecolor="white")
axes[0].axvline(df["aqi"].mean(), color="red", linestyle="--", label=f"Mean: {df['aqi'].mean():.1f}")
axes[0].set_title("AQI distribution")
axes[0].set_xlabel("AQI")
axes[0].legend()

# AQI category breakdown
bins   = [0, 50, 100, 150, 200, 300, 500]
labels = ["Good", "Moderate", "Unhealthy-Sens.", "Unhealthy", "Very Unhealthy", "Hazardous"]
df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=labels)
counts = df["aqi_category"].value_counts().reindex(labels)
axes[1].bar(counts.index, counts.values,
            color=["green","gold","orange","red","purple","maroon"])
axes[1].set_title("AQI category breakdown")
axes[1].set_xlabel("Category")
axes[1].set_ylabel("Hours")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("eda_distribution.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Temporal patterns: hour of day and day of week

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

hourly = df.groupby("hour")["aqi"].mean()
axes[0].bar(hourly.index, hourly.values, color="steelblue")
axes[0].set_title("Average AQI by hour of day")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Mean AQI")
axes[0].set_xticks(range(0, 24, 2))

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily = df.groupby("day_of_week")["aqi"].mean()
axes[1].bar([day_names[i] for i in daily.index], daily.values, color="coral")
axes[1].set_title("Average AQI by day of week")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Mean AQI")
plt.tight_layout()
plt.savefig("eda_temporal.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Feature correlation heatmap

# %%
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()

plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, annot_kws={"size": 8}
)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=120)
plt.show()

# %% [markdown]
# ## 5. AQI vs weather features

# %%
weather_cols = ["temp", "humidity", "pressure", "wind_speed"]
weather_cols = [c for c in weather_cols if c in df.columns]

fig, axes = plt.subplots(1, len(weather_cols), figsize=(5 * len(weather_cols), 4))
if len(weather_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, weather_cols):
    ax.scatter(df[col], df["aqi"], alpha=0.3, s=10, color="steelblue")
    # Trend line
    z = np.polyfit(df[col].dropna(), df.loc[df[col].notna(), "aqi"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xs, p(xs), color="red", linewidth=1.5)
    r = df[[col, "aqi"]].dropna().corr().iloc[0, 1]
    ax.set_title(f"AQI vs {col}\n(r = {r:.2f})")
    ax.set_xlabel(col)
    ax.set_ylabel("AQI")

plt.tight_layout()
plt.savefig("eda_weather_scatter.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Lag feature analysis: autocorrelation

# %%
from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF manually for first 48 lags
lags = range(1, 49)
acf_vals = [df["aqi"].autocorr(lag=l) for l in lags]
axes[0].bar(lags, acf_vals, color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].axhline(0.2,  color="red", linestyle="--", linewidth=0.8, label="±0.2 threshold")
axes[0].axhline(-0.2, color="red", linestyle="--", linewidth=0.8)
axes[0].set_title("Autocorrelation of AQI (lags 1–48h)")
axes[0].set_xlabel("Lag (hours)")
axes[0].set_ylabel("Autocorrelation")
axes[0].legend()

# Lag scatter: AQI(t) vs AQI(t-1)
axes[1].scatter(df["aqi"].shift(1), df["aqi"], alpha=0.3, s=8, color="coral")
axes[1].set_title("AQI(t) vs AQI(t-1)")
axes[1].set_xlabel("AQI at t-1")
axes[1].set_ylabel("AQI at t")

plt.tight_layout()
plt.savefig("eda_autocorr.png", dpi=120)
plt.show()

# %% [markdown]
# ## 7. Rolling mean: trend smoothing

# %%
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["timestamp"], df["aqi"],
        color="lightsteelblue", linewidth=0.5, label="Raw AQI", alpha=0.7)
ax.plot(df["timestamp"], df["aqi_rolling_24h_mean"],
        color="steelblue", linewidth=1.5, label="24h rolling mean")
ax.plot(df["timestamp"], df["aqi_rolling_6h_mean"],
        color="coral", linewidth=1.0, linestyle="--", label="6h rolling mean")
ax.set_title("AQI with rolling averages")
ax.set_ylabel("AQI")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_rolling.png", dpi=120)
plt.show()

# %% [markdown]
# ## 8. Summary statistics

# %%
summary = {
    "Total hours":          len(df),
    "Mean AQI":             round(df["aqi"].mean(), 1),
    "Median AQI":           round(df["aqi"].median(), 1),
    "Max AQI":              df["aqi"].max(),
    "Min AQI":              df["aqi"].min(),
    "Hours > 100 (Unhealthy)": (df["aqi"] > 100).sum(),
    "Hours > 150 (Very Unhealthy)": (df["aqi"] > 150).sum(),
    "Peak hour of day":     int(df.groupby("hour")["aqi"].mean().idxmax()),
    "Cleanest hour of day": int(df.groupby("hour")["aqi"].mean().idxmin()),
}
for k, v in summary.items():
    print(f"  {k:<35} {v}")
