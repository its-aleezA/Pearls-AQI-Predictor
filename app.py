import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Rawalpindi AQI Predictor",
    page_icon="🌤️",
    layout="wide",
)

# Palette ───────────────────────────────────────────────────────────────────
LIGHT_BLUE  = "#B8D8D8"
COOL_STEEL  = "#7A9E9F"
BLUE_SLATE  = "#4F6367"
BEIGE       = "#EEF5DB"
CORAL       = "#FE5F55"

# Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* Global reset */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {BLUE_SLATE};
}}

/* Page background */
.stApp {{
    background-color: {BEIGE};
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 0rem !important; }}

/* Hero banner */
.hero {{
    background: linear-gradient(160deg, {BLUE_SLATE} 0%, {COOL_STEEL} 60%, {LIGHT_BLUE} 100%);
    border-radius: 0 0 32px 32px;
    padding: 2.5rem 3rem 0 3rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}}
.hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: {BEIGE};
    line-height: 1.1;
    margin: 0;
}}
.hero-sub {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 300;
    color: {LIGHT_BLUE};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.hero-desc {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: {LIGHT_BLUE};
    opacity: 0.85;
    max-width: 520px;
    margin-top: 0.5rem;
    padding-bottom: 1.5rem;
}}

/* Metric cards */
.metric-card {{
    background: white;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid {COOL_STEEL};
    box-shadow: 0 2px 12px rgba(79,99,103,0.08);
}}
.metric-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {COOL_STEEL};
    font-weight: 500;
    margin-bottom: 0.3rem;
}}
.metric-value {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: {BLUE_SLATE};
    line-height: 1;
}}
.metric-card.accent {{ border-left-color: {CORAL}; }}
.metric-card.accent .metric-value {{ color: {CORAL}; }}

/* Alert banners */
.alert-box {{
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    font-weight: 500;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}}
.alert-hazardous {{
    background: #fff0ef;
    border: 1.5px solid {CORAL};
    color: #c0392b;
}}
.alert-warning {{
    background: #fff8f0;
    border: 1.5px solid #f0a500;
    color: #8a5e00;
}}
.alert-good {{
    background: #f0faf4;
    border: 1.5px solid {COOL_STEEL};
    color: {BLUE_SLATE};
}}

/* Day cards */
.day-card {{
    background: white;
    border-radius: 16px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(79,99,103,0.07);
    height: 100%;
}}
.day-card-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {COOL_STEEL};
    margin-bottom: 0.5rem;
}}
.day-card-value {{
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: {BLUE_SLATE};
    font-weight: 700;
    line-height: 1.2;
}}
.day-card-cat {{
    font-size: 0.8rem;
    color: {COOL_STEEL};
    margin-top: 0.4rem;
    font-style: italic;
}}

/* Section headers */
.section-heading {{
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: {BLUE_SLATE};
    margin: 1.8rem 0 0.8rem 0;
    font-weight: 700;
}}
.section-rule {{
    border: none;
    border-top: 1.5px solid {LIGHT_BLUE};
    margin: 0.5rem 0 1.2rem 0;
}}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid {LIGHT_BLUE};
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    color: {COOL_STEEL};
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
}}
.stTabs [aria-selected="true"] {{
    background: {BLUE_SLATE} !important;
    color: {BEIGE} !important;
}}

/* Dataframe */
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

/* Download buttons */
.stDownloadButton button {{
    background: {BLUE_SLATE};
    color: {BEIGE};
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
}}
.stDownloadButton button:hover {{
    background: {COOL_STEEL};
    color: white;
}}

/* Chart area */
.element-container iframe {{ border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

# AQI helpers
def aqi_category(aqi: float) -> tuple:
    if aqi <= 50:   return "Good", COOL_STEEL
    elif aqi <= 100: return "Moderate", "#f0a500"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#e07b00"
    elif aqi <= 200: return "Unhealthy", CORAL
    elif aqi <= 300: return "Very Unhealthy", "#c0392b"
    else:            return "Hazardous", "#8e1a12"

def aqi_dot(aqi: float) -> str:
    _, color = aqi_category(aqi)
    return f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px;"></span>'

# Skyline SVG
def skyline_svg(aqi: float) -> str:
    if aqi <= 100:   sky1, sky2, haze = "#B8D8D8", "#7A9E9F", "rgba(184,216,216,0)"
    elif aqi <= 200: sky1, sky2, haze = "#d4b896", "#c49a6c", "rgba(210,160,80,0.25)"
    elif aqi <= 300: sky1, sky2, haze = "#c4956a", "#a06040", "rgba(180,100,50,0.35)"
    else:            sky1, sky2, haze = "#9b6b6b", "#7a3f3f", "rgba(150,60,60,0.45)"

    return f"""
    <svg viewBox="0 0 900 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;margin-top:-4px;">
      <defs>
        <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{sky1}"/>
          <stop offset="100%" stop2-color="{sky2}" stop-color="{sky2}"/>
        </linearGradient>
        <linearGradient id="haze" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{haze}"/>
          <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
        </linearGradient>
      </defs>
      <!-- sky -->
      <rect width="900" height="180" fill="url(#sky)"/>
      <!-- haze overlay -->
      <rect width="900" height="180" fill="url(#haze)"/>
      <!-- buildings — silhouette of Rawalpindi/Islamabad skyline -->
      <g fill="{BLUE_SLATE}" opacity="0.85">
        <!-- Faisal Mosque dome hint -->
        <polygon points="60,80 75,40 90,80"/>
        <rect x="68" y="80" width="14" height="60"/>
        <!-- tall office block -->
        <rect x="100" y="55" width="30" height="85"/>
        <rect x="108" y="48" width="14" height="10"/>
        <!-- antenna -->
        <rect x="114" y="30" width="2" height="20"/>
        <!-- mid blocks -->
        <rect x="145" y="75" width="22" height="65"/>
        <rect x="172" y="60" width="18" height="80"/>
        <rect x="196" y="82" width="14" height="58"/>
        <!-- stepped tower -->
        <rect x="220" y="50" width="28" height="90"/>
        <rect x="224" y="42" width="20" height="12"/>
        <rect x="228" y="34" width="12" height="10"/>
        <rect x="233" y="26" width="2" height="10"/>
        <!-- low blocks -->
        <rect x="256" y="90" width="20" height="50"/>
        <rect x="282" y="70" width="24" height="70"/>
        <rect x="312" y="85" width="16" height="55"/>
        <!-- wide block -->
        <rect x="334" y="65" width="40" height="75"/>
        <rect x="344" y="58" width="20" height="10"/>
        <!-- thin tower -->
        <rect x="382" y="40" width="10" height="100"/>
        <rect x="379" y="36" width="16" height="6"/>
        <rect x="386" y="20" width="2" height="18"/>
        <!-- right cluster -->
        <rect x="400" y="78" width="22" height="62"/>
        <rect x="428" y="62" width="26" height="78"/>
        <rect x="460" y="88" width="18" height="52"/>
        <rect x="484" y="55" width="32" height="85"/>
        <rect x="492" y="48" width="16" height="10"/>
        <!-- minaret pair -->
        <rect x="525" y="50" width="6" height="90"/>
        <rect x="545" y="50" width="6" height="90"/>
        <ellipse cx="528" cy="50" rx="5" ry="8"/>
        <ellipse cx="548" cy="50" rx="5" ry="8"/>
        <!-- bridge/overpass hint -->
        <rect x="560" y="95" width="80" height="8" rx="4"/>
        <rect x="572" y="95" width="6" height="45"/>
        <rect x="622" y="95" width="6" height="45"/>
        <!-- far right blocks -->
        <rect x="648" y="72" width="24" height="68"/>
        <rect x="678" y="58" width="20" height="82"/>
        <rect x="704" y="80" width="16" height="60"/>
        <rect x="726" y="50" width="28" height="90"/>
        <rect x="734" y="42" width="12" height="10"/>
        <rect x="739" y="28" width="2" height="16"/>
        <rect x="760" y="82" width="20" height="58"/>
        <rect x="786" y="68" width="22" height="72"/>
        <rect x="814" y="55" width="18" height="85"/>
        <rect x="838" y="78" width="16" height="62"/>
        <rect x="860" y="45" width="30" height="95"/>
        <rect x="868" y="38" width="14" height="10"/>
      </g>
      <!-- ground -->
      <rect x="0" y="138" width="900" height="42" fill="{BLUE_SLATE}" opacity="0.9"/>
    </svg>"""

# File paths
VALIDATION_CSV = "predictions/aqi_batch_predictions.csv"
FORECAST_CSV   = "predictions/aqi_72h_forecast.csv"
SHAP_PLOT      = "models/shap_summary.png"

# Load data
missing = [p for p in [VALIDATION_CSV, FORECAST_CSV] if not os.path.exists(p)]
if missing:
    st.error(f"❌ Missing prediction files: {missing}. Run `python batch_inference.py` first.")
    st.stop()

df_val      = pd.read_csv(VALIDATION_CSV)
df_forecast = pd.read_csv(FORECAST_CSV)
max_aqi     = df_forecast["Forecast_AQI"].max()
next_1h     = df_forecast["Forecast_AQI"].iloc[0]

# Hero 
st.markdown(f"""
<div class="hero">
  <div class="hero-sub">Rawalpindi · Live Air Quality Intelligence</div>
  <div class="hero-title">AQI Forecast<br><em style="font-style:italic;font-weight:400;">Dashboard</em></div>
  <div class="hero-desc">
    72-hour forward predictions · Hopsworks Feature Store · GitHub Actions CI/CD
  </div>
</div>
""", unsafe_allow_html=True)

# Render skyline SVG via components.html
_sky_html = f"""
<div style="margin:-1rem -1rem 2rem -1rem;overflow:hidden;border-radius:0 0 32px 32px;">
  {skyline_svg(next_1h)}
</div>
"""
components.html(_sky_html, height=185, scrolling=False)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "72-Hour Forecast",
    "Model Validation",
    "Explainability",
    "Raw Data",
])

# Tab 1 : Live Forecast
with tab1:
    cat_label, cat_color = aqi_category(max_aqi)

    # Alert banner
    if max_aqi > 300:
        st.markdown(f"""<div class="alert-box alert-hazardous">
            🚨 <div><strong>Hazardous AQI forecasted in the next 72 hours.</strong><br>
            Avoid all outdoor activity. Wear N95 masks even indoors if possible.</div>
        </div>""", unsafe_allow_html=True)
    elif max_aqi > 200:
        st.markdown(f"""<div class="alert-box alert-warning">
            ⚠️ <div><strong>Very Unhealthy AQI forecasted.</strong><br>
            The general public should avoid prolonged outdoor exposure.</div>
        </div>""", unsafe_allow_html=True)
    elif max_aqi > 150:
        st.markdown(f"""<div class="alert-box alert-warning">
            ⚠️ <div><strong>Unhealthy AQI forecasted.</strong><br>
            Children, elderly, and those with respiratory conditions should limit outdoor time.</div>
        </div>""", unsafe_allow_html=True)
    elif max_aqi > 100:
        st.markdown(f"""<div class="alert-box alert-warning">
            🟡 <div><strong>Moderate AQI forecasted.</strong><br>
            Unusually sensitive individuals should reduce prolonged outdoor exertion.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert-box alert-good">
            ✅ <div><strong>Good air quality forecasted for the next 72 hours.</strong><br>
            Enjoy outdoor activities : Rawalpindi's skies are clear.</div>
        </div>""", unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card accent">
            <div class="metric-label">Next 1h AQI</div>
            <div class="metric-value">{next_1h}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        peak_24 = df_forecast["Forecast_AQI"].iloc[:24].max()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">24h Peak</div>
            <div class="metric-value">{peak_24}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">72h Peak</div>
            <div class="metric-value">{max_aqi}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_72 = round(df_forecast["Forecast_AQI"].mean(), 1)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">72h Average</div>
            <div class="metric-value">{avg_72}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast chart
    st.markdown('<div class="section-heading">Forecast Timeline</div><hr class="section-rule">', unsafe_allow_html=True)
    plot_df = df_forecast.set_index("Timestamp")[["Forecast_AQI"]]
    st.line_chart(plot_df, height=260, color=CORAL)

    # Day cards
    st.markdown('<div class="section-heading">Day-by-Day Outlook</div><hr class="section-rule">', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for col, (day_label, start, end) in zip([d1, d2, d3], [
        ("Day 1 : Next 24h", 0, 24),
        ("Day 2", 24, 48),
        ("Day 3", 48, 72),
    ]):
        chunk = df_forecast["Forecast_AQI"].iloc[start:end]
        cat, color = aqi_category(chunk.mean())
        with col:
            st.markdown(f"""<div class="day-card">
                <div class="day-card-label">{day_label}</div>
                <div class="day-card-value"
                     style="color:{color}">
                    Avg {chunk.mean():.0f}
                </div>
                <div style="font-size:0.85rem;color:{BLUE_SLATE};margin-top:2px;">
                    Peak {chunk.max():.0f}
                </div>
                <div class="day-card-cat">{cat}</div>
            </div>""", unsafe_allow_html=True)

# Tab 2 : Model Validation
with tab2:
    st.markdown('<div class="section-heading">Historical: Actual vs Predicted AQI</div><hr class="section-rule">', unsafe_allow_html=True)

    latest_actual = round(df_val["Actual_AQI"].iloc[-1], 1)
    latest_pred   = round(df_val["Predicted_AQI"].iloc[-1], 1)
    error         = round(latest_actual - latest_pred, 1)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Latest Actual AQI</div>
            <div class="metric-value">{latest_actual}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card accent">
            <div class="metric-label">Latest Predicted AQI</div>
            <div class="metric-value">{latest_pred}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        err_color = CORAL if abs(error) > 20 else COOL_STEEL
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Prediction Error</div>
            <div class="metric-value" style="color:{err_color}">{error:+.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    plot_df = df_val.tail(100).copy().set_index("Timestamp")
    st.line_chart(plot_df[["Actual_AQI", "Predicted_AQI"]], height=300,
                  color=[BLUE_SLATE, CORAL])

    if os.path.exists("predictions/aqi_inference_plot.png"):
        st.markdown('<div class="section-heading">Monitoring Plot</div><hr class="section-rule">', unsafe_allow_html=True)
        st.image("predictions/aqi_inference_plot.png", use_container_width=True)

# Tab 3 : Explainability
with tab3:
    st.markdown('<div class="section-heading">Feature Importance : SHAP</div><hr class="section-rule">', unsafe_allow_html=True)
    st.markdown(f"""
    <p style="color:{COOL_STEEL};font-size:0.95rem;max-width:600px;line-height:1.7;">
    SHAP (SHapley Additive exPlanations) quantifies each feature's contribution
    to individual predictions. Features at the top drive the model's output most.
    Red = pushes AQI higher · Blue = pushes AQI lower.
    </p>
    """, unsafe_allow_html=True)
    if os.path.exists(SHAP_PLOT):
        st.image(SHAP_PLOT, use_container_width=True)
    else:
        st.markdown(f"""<div class="alert-box alert-good">
            ℹ️ SHAP plot not found. Run <code>python train_aqi_model.py</code> to generate
            <code>models/shap_summary.png</code>.
        </div>""", unsafe_allow_html=True)

# Tab 4 : Raw Data
with tab4:
    st.markdown('<div class="section-heading">Validation Logs</div><hr class="section-rule">', unsafe_allow_html=True)
    st.dataframe(df_val, use_container_width=True, height=280)

    st.markdown('<div class="section-heading">72-Hour Forecast Table</div><hr class="section-rule">', unsafe_allow_html=True)
    st.dataframe(df_forecast, use_container_width=True, height=280)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇ Download Validation CSV",
            data=df_val.to_csv(index=False),
            file_name="aqi_validation.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇ Download Forecast CSV",
            data=df_forecast.to_csv(index=False),
            file_name="aqi_72h_forecast.csv",
            mime="text/csv",
        )
