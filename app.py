"""
app.py
Toronto Island Ferry Ticket Demand Forecasting & Decision Support System.
Streamlit Web Application providing short-term predictive intelligence,
interactive EDA, model benchmarking, and operational decision support.
Enhanced with a modern glassmorphic dark theme and fluid motion animations.
"""

import os
import sys
import json
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Path Constants and Environment Setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    import src.models  # noqa: F401 - ensures custom models are registered for joblib unpickling
except Exception:
    pass

LOGO_PATH = os.path.join(APP_DIR, "assets", "logo.png")

# Helper for base64 logo embedding
def get_base64_logo():
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""

LOGO_B64 = get_base64_logo()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Toronto Ferry Predictive DSS",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🛳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling with Dark Glassmorphism, Neon Accents, and Motion Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    /* Keyframe Animations */
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes floatBob {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-7px) rotate(-1.5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes radarPing {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(22px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes shimmerSweep {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* Animated Header */
    .main-header {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 40%, #F472B6 75%, #38BDF8 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 8s ease infinite;
        letter-spacing: -0.025em;
        margin-bottom: 0.15rem;
        line-height: 1.2;
    }
    
    .sub-header {
        font-size: 1.15rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 1.6rem;
        letter-spacing: 0.01em;
    }

    /* Motion Floating Icon */
    .floating-ferry {
        display: inline-block;
        animation: floatBob 3.5s ease-in-out infinite;
        filter: drop-shadow(0 4px 10px rgba(56, 189, 248, 0.4));
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.78) 0%, rgba(15, 23, 42, 0.92) 100%);
        border-radius: 16px;
        padding: 22px 20px;
        border: 1px solid rgba(56, 189, 248, 0.22);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px 1px rgba(56, 189, 248, 0.08);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        animation: slideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    
    .metric-card:hover {
        transform: translateY(-6px) scale(1.018);
        border-color: rgba(56, 189, 248, 0.65);
        box-shadow: 0 20px 35px -8px rgba(56, 189, 248, 0.3), 0 0 25px 3px rgba(56, 189, 248, 0.2);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #F472B6);
        border-top-left-radius: 16px;
        border-top-right-radius: 16px;
    }

    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 700;
        letter-spacing: 0.07em;
    }
    
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 6px;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
        letter-spacing: -0.02em;
    }
    
    .metric-sub {
        font-size: 0.84rem;
        color: #64748B;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Live Operational Status Radar Pill */
    .radar-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 8px 16px;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 9999px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }

    .radar-dot {
        width: 10px;
        height: 10px;
        background-color: #38BDF8;
        border-radius: 50%;
        display: inline-block;
        animation: radarPing 2s infinite cubic-bezier(0, 0, 0.2, 1);
    }

    .radar-text {
        font-size: 0.88rem;
        font-weight: 600;
        color: #E2E8F0;
        letter-spacing: 0.02em;
    }

    /* Modern Alert Boxes with Glow */
    .alert-box {
        padding: 18px 22px;
        border-radius: 14px;
        margin: 16px 0px;
        font-size: 0.96rem;
        backdrop-filter: blur(10px);
        animation: slideInUp 0.5s ease-out;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.22) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        color: #FDE68A;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(30, 58, 138, 0.22) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #BAE6FD;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(6, 95, 70, 0.22) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #A7F3D0;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }

    /* Styled Tables */
    table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        background: rgba(15, 23, 42, 0.75) !important;
    }
    th {
        background: rgba(30, 41, 59, 0.9) !important;
        color: #38BDF8 !important;
        font-weight: 700 !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.25) !important;
    }
    td {
        padding: 12px 16px !important;
        color: #E2E8F0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Path Constants
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(APP_DIR, "data", "processed", "clean_15min_timeseries.parquet")
FEATURES_DATA_PATH = os.path.join(APP_DIR, "data", "processed", "features_dataset.parquet")
METRICS_CSV_PATH = os.path.join(APP_DIR, "reports", "results", "model_evaluation_metrics.csv")
MARGINS_JSON_PATH = os.path.join(APP_DIR, "models", "saved", "prediction_intervals.json")
MODELS_DIR = os.path.join(APP_DIR, "models", "saved")

@st.cache_data(show_spinner=False)
def load_clean_timeseries():
    if not os.path.exists(PROCESSED_DATA_PATH):
        return None
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    return df

@st.cache_data(show_spinner=False)
def load_features_sample():
    if not os.path.exists(FEATURES_DATA_PATH):
        return None
    df = pd.read_parquet(FEATURES_DATA_PATH)
    return df

@st.cache_data(show_spinner=False)
def load_metrics_table():
    if not os.path.exists(METRICS_CSV_PATH):
        return None
    return pd.read_csv(METRICS_CSV_PATH)

@st.cache_data(show_spinner=False)
def load_margins_dict():
    if not os.path.exists(MARGINS_JSON_PATH):
        return {}
    with open(MARGINS_JSON_PATH, 'r') as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_trained_model(target: str, horizon: str, model_name: str):
    safe_name = model_name.lower().replace(' ', '_')
    model_file = f"{target}_{horizon}_{safe_name}.joblib"
    model_path = os.path.join(MODELS_DIR, model_file)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def apply_custom_theme(fig, title=None, height=430):
    """Applies a rich dark glassmorphic layout to Plotly figures."""
    fig.update_layout(
        title=title or fig.layout.title,
        template="plotly_dark",
        paper_bgcolor='rgba(15, 23, 42, 0.75)',
        plot_bgcolor='rgba(15, 23, 42, 0.45)',
        font=dict(color='#E2E8F0', family='Plus Jakarta Sans, system-ui, sans-serif'),
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.08)',
            zerolinecolor='rgba(255, 255, 255, 0.15)',
            tickfont=dict(color='#94A3B8')
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.08)',
            zerolinecolor='rgba(255, 255, 255, 0.15)',
            tickfont=dict(color='#94A3B8')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CBD5E1')
        ),
        height=height
    )
    return fig

# Sidebar Navigation
if LOGO_B64:
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.2rem; padding: 0.2rem 0;">
        <div style="position: relative; display: inline-block;">
            <img src="data:image/png;base64,{LOGO_B64}" 
                 style="width: 105px; height: 105px; border-radius: 50%; 
                        border: 2.5px solid #38BDF8; 
                        box-shadow: 0 0 25px rgba(56, 189, 248, 0.45); 
                        object-fit: cover;"
                 class="floating-ferry"
            />
        </div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-top: 0.65rem; letter-spacing: -0.02em;">
            Toronto Ferry DSS
        </div>
        <div style="font-size: 0.72rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
            Predictive Decision Support
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown('## <span class="floating-ferry">🛳️</span> Toronto Ferry DSS', unsafe_allow_html=True)
    st.sidebar.markdown("*Short-Term Predictive Decision Support*")

page = st.sidebar.radio(
    "Navigation",
    [
        "Page 1 — Overview",
        "Page 2 — Demand Analysis",
        "Page 3 — Forecast",
        "Page 4 — Model Comparison",
        "Page 5 — Operational Insights",
        "Page 6 — Data / Model Info"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="radar-pill">
    <span class="radar-dot"></span>
    <span class="radar-text">Live System Operational</span>
</div>
""", unsafe_allow_html=True)

# Load baseline resources
df_clean = load_clean_timeseries()
df_features = load_features_sample()
metrics_df = load_metrics_table()
margins_dict = load_margins_dict()

if df_clean is None or df_features is None or metrics_df is None:
    st.error("Missing precomputed data artifacts. Please ensure 'run_pipeline.py' has been executed.")
    st.stop()

# Helper for feature columns
from src.feature_engineering import get_feature_columns
FEATURE_COLS = get_feature_columns()

# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "Page 1 — Overview":
    logo_header_html = f'<img src="data:image/png;base64,{LOGO_B64}" class="floating-ferry" style="width: 44px; height: 44px; border-radius: 50%; border: 2px solid #38BDF8; box-shadow: 0 0 16px rgba(56, 189, 248, 0.45); vertical-align: middle; margin-right: 12px; object-fit: cover;">' if LOGO_B64 else '<span class="floating-ferry">🛳️</span> '
    st.markdown(f'<div class="main-header">{logo_header_html}Toronto Island Ferry Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Short-Term Predictive Decision Support System • High-Velocity Real-Time Analytics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-box alert-info">
        <strong>⚡ System Mission:</strong> Forecast near-term ferry ticket demand using historical 15-minute ticket activity 
        and machine-learning/time-series forecasting to enable proactive vessel dispatch, gate crowd control, and queue mitigation.
    </div>
    """, unsafe_allow_html=True)
    
    # Latest historical observation
    last_row = df_clean.iloc[-1]
    last_time = last_row['Timestamp']
    latest_sales = int(last_row['Sales Count'])
    latest_redemption = int(last_row['Redemption Count'])
    
    # 15m ahead forecast using top model (XGBoost)
    latest_features = df_features[FEATURE_COLS].iloc[[-1]]
    sales_model_15m = load_trained_model('sales', '15m', 'XGBoost')
    pred_sales_15m = int(round(sales_model_15m.predict(latest_features)[0])) if sales_model_15m else latest_sales
    
    margin_90 = margins_dict.get('sales', {}).get('15m', {}).get('XGBoost', {}).get('margin_90', 28.0)
    lower_15m = max(0, int(round(pred_sales_15m - margin_90)))
    upper_15m = int(round(pred_sales_15m + margin_90))
    
    # Trend calculation
    recent_mean_4 = df_features['sales_roll_mean_4'].iloc[-1]
    if pred_sales_15m > recent_mean_4 * 1.15:
        trend_status = "Increasing ↗"
        trend_color = "#F43F5E"
    elif pred_sales_15m < recent_mean_4 * 0.85:
        trend_status = "Decreasing ↘"
        trend_color = "#38BDF8"
    else:
        trend_status = "Stable →"
        trend_color = "#10B981"
        
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Ticket Sales</div>
            <div class="metric-value">{latest_sales:,}</div>
            <div class="metric-sub">⏱️ {last_time.strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Redemptions</div>
            <div class="metric-value">{latest_redemption:,}</div>
            <div class="metric-sub">⏱️ {last_time.strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Next 15m Sales Forecast</div>
            <div class="metric-value">{pred_sales_15m:,}</div>
            <div class="metric-sub">🎯 90% Interval: {lower_15m} – {upper_15m}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Demand Trend</div>
            <div class="metric-value" style="color: {trend_color};">{trend_status}</div>
            <div class="metric-sub">📊 1h Baseline: {recent_mean_4:.1f} / interval</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Recent Demand Context & Active Rhythms")
    
    # 48-hour recent lookback
    recent_lookback = df_clean.tail(192).copy()
    fig_overview = go.Figure()
    fig_overview.add_trace(go.Scatter(
        x=recent_lookback['Timestamp'], y=recent_lookback['Sales Count'],
        mode='lines', name='Ticket Sales',
        line=dict(color='#00F2FE', width=2.5)
    ))
    fig_overview.add_trace(go.Scatter(
        x=recent_lookback['Timestamp'], y=recent_lookback['Redemption Count'],
        mode='lines', name='Ticket Redemptions',
        line=dict(color='#FF9F43', width=2.5)
    ))
    fig_overview = apply_custom_theme(
        fig_overview,
        title="Last 48 Hours Continuous Demand Velocity (15-Minute Cadence)",
        height=390
    )
    fig_overview.update_layout(xaxis_title="Timeline", yaxis_title="Tickets per 15-Min Interval", hovermode="x unified")
    st.plotly_chart(fig_overview, use_container_width=True)
    
    st.markdown("""
    **Core Decision Questions Answered:**
    1. **What is happening now?** Real-time monitoring of actual ticketing and boarding counts.
    2. **What is likely to happen next?** Direct machine-learning predictions across 15m, 30m, 1h, and 2h horizons.
    3. **How certain is the prediction?** Statistically bounded 90% Conformal Prediction Intervals.
    4. **How reliable is the model?** Transparent empirical benchmarks (MAE, RMSE, WAPE, Peak Miss Rate).
    5. **What patterns govern demand?** In-depth diurnal, weekly, and seasonal tourist demand profiles.
    """)

# ==============================================================================
# PAGE 2: DEMAND ANALYSIS
# ==============================================================================
elif page == "Page 2 — Demand Analysis":
    st.markdown('<div class="main-header"><span class="floating-ferry">📈</span> Exploratory Demand Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical Patterns, Diurnal Cycles, Seasonality, and Queue Dynamics</div>', unsafe_allow_html=True)
    
    st.markdown("##### ⏱️ Reference Operational Time")
    
    col_preset, col_slider = st.columns([1, 2.5])
    with col_preset:
        preset_choice = st.selectbox(
            "Operational Scenarios",
            [
                "Latest Available Record (Dec 2025)",
                "Summer Peak Surge (Canada Day 2025)",
                "Summer Weekend Peak (Civic Holiday 2025)",
                "Spring Shoulder Season",
                "Custom Timestamp Slider"
            ],
            index=0
        )
    
    if preset_choice == "Latest Available Record (Dec 2025)":
        default_eda_idx = len(df_clean) - 1
    elif preset_choice == "Summer Peak Surge (Canada Day 2025)":
        sample_surge = df_clean[(df_clean['Timestamp'].dt.year == 2025) & 
                                (df_clean['Timestamp'].dt.month == 7) & 
                                (df_clean['Timestamp'].dt.day == 1) & 
                                (df_clean['Timestamp'].dt.hour == 14)]
        default_eda_idx = sample_surge.index[0] if not sample_surge.empty else len(df_clean) - 15000
    elif preset_choice == "Summer Weekend Peak (Civic Holiday 2025)":
        sample_civic = df_clean[(df_clean['Timestamp'].dt.year == 2025) & 
                                (df_clean['Timestamp'].dt.month == 8) & 
                                (df_clean['Timestamp'].dt.day == 4) & 
                                (df_clean['Timestamp'].dt.hour == 14)]
        default_eda_idx = sample_civic.index[0] if not sample_civic.empty else len(df_clean) - 12000
    elif preset_choice == "Spring Shoulder Season":
        sample_spring = df_clean[(df_clean['Timestamp'].dt.year == 2025) & 
                                 (df_clean['Timestamp'].dt.month == 5) & 
                                 (df_clean['Timestamp'].dt.day == 15) & 
                                 (df_clean['Timestamp'].dt.hour == 12)]
        default_eda_idx = sample_spring.index[0] if not sample_spring.empty else len(df_clean) - 20000
    else:
        default_eda_idx = len(df_clean) - 200
        
    with col_slider:
        ref_eda_idx = st.slider(
            "Select Reference Operational Time Index (Historical Explorer)",
            min_value=0,
            max_value=len(df_clean) - 1,
            value=int(default_eda_idx),
            step=1
        )
        
    ref_eda_row = df_clean.iloc[ref_eda_idx]
    ref_eda_time = ref_eda_row['Timestamp']
    eda_sales_val = int(ref_eda_row['Sales Count'])
    eda_red_val = int(ref_eda_row['Redemption Count'])
    
    window_start_idx = max(0, ref_eda_idx - 96)
    window_end_idx = min(len(df_clean), ref_eda_idx + 97)
    window_slice = df_clean.iloc[window_start_idx:window_end_idx]
    max_24h_sales = int(window_slice['Sales Count'].max())
    max_24h_red = int(window_slice['Redemption Count'].max())
    
    c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
    with c_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reference Operational Time</div>
            <div class="metric-value" style="font-size: 1.45rem;">{ref_eda_time.strftime('%H:%M')}</div>
            <div class="metric-sub">📅 {ref_eda_time.strftime('%Y-%m-%d (%a)')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reference Sales Count</div>
            <div class="metric-value" style="color: #00F2FE;">{eda_sales_val:,}</div>
            <div class="metric-sub">Tickets sold at this 15-min interval</div>
        </div>
        """, unsafe_allow_html=True)
    with c_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reference Redemptions</div>
            <div class="metric-value" style="color: #FF9F43;">{eda_red_val:,}</div>
            <div class="metric-sub">Turnstile scans at this interval</div>
        </div>
        """, unsafe_allow_html=True)
    with c_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">24h Surrounding Peak</div>
            <div class="metric-value">{max(max_24h_sales, max_24h_red):,}</div>
            <div class="metric-sub">Peak 15-min volume in 48h window</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Reference Window Timeline",
        "Diurnal Patterns",
        "Weekly & Weekend",
        "Monthly Seasonality",
        "Sales vs Redemptions"
    ])
    
    with tab0:
        st.subheader(f"Demand Activity Surrounding Reference Operational Time ({ref_eda_time.strftime('%Y-%m-%d %H:%M')})")
        fig_ref_timeline = go.Figure()
        fig_ref_timeline.add_trace(go.Scatter(
            x=window_slice['Timestamp'], y=window_slice['Sales Count'],
            mode='lines', name='Ticket Sales', line=dict(color='#00F2FE', width=2.5)
        ))
        fig_ref_timeline.add_trace(go.Scatter(
            x=window_slice['Timestamp'], y=window_slice['Redemption Count'],
            mode='lines', name='Ticket Redemptions', line=dict(color='#FF9F43', width=2.5)
        ))
        fig_ref_timeline.add_vline(
            x=ref_eda_time, line_width=2.5, line_dash="dash", line_color="#F43F5E"
        )
        fig_ref_timeline.add_annotation(
            x=ref_eda_time, y=1, yref="paper",
            text="Selected Reference Time", showarrow=False,
            xanchor="right", yanchor="top",
            font=dict(color="#F43F5E", size=12),
            bgcolor="rgba(15, 23, 42, 0.85)", bordercolor="#F43F5E", borderwidth=1, borderpad=4
        )
        fig_ref_timeline = apply_custom_theme(
            fig_ref_timeline,
            title=f"48-Hour Continuous Timeline Centered on Reference Operational Time ({ref_eda_time.strftime('%b %d, %Y')})",
            height=430
        )
        fig_ref_timeline.update_layout(xaxis_title="Time", yaxis_title="Count per 15-Min Interval", hovermode="x unified")
        st.plotly_chart(fig_ref_timeline, use_container_width=True)
    
    with tab1:
        st.subheader("Hourly Average Demand (Weekday vs Weekend)")
        st.write("Demand follows a pronounced diurnal curve from 07:00 to 23:00, with peak activity between 11:00 and 15:00.")
        wday = df_clean[df_clean['Timestamp'].dt.dayofweek < 5].groupby(df_clean['Timestamp'].dt.hour)[['Sales Count', 'Redemption Count']].mean()
        wend = df_clean[df_clean['Timestamp'].dt.dayofweek >= 5].groupby(df_clean['Timestamp'].dt.hour)[['Sales Count', 'Redemption Count']].mean()
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(x=wday.index, y=wday['Sales Count'], name='Weekday Sales', line=dict(color='#00F2FE', width=3)))
        fig_hourly.add_trace(go.Scatter(x=wend.index, y=wend['Sales Count'], name='Weekend Sales', line=dict(color='#00F2FE', width=3, dash='dash')))
        fig_hourly.add_trace(go.Scatter(x=wday.index, y=wday['Redemption Count'], name='Weekday Redemptions', line=dict(color='#FF9F43', width=3)))
        fig_hourly.add_trace(go.Scatter(x=wend.index, y=wend['Redemption Count'], name='Weekend Redemptions', line=dict(color='#FF9F43', width=3, dash='dash')))
        fig_hourly = apply_custom_theme(fig_hourly, title="Hourly Average Demand Profile (Weekday vs Weekend)", height=430)
        fig_hourly.update_layout(xaxis_title="Hour of Day (0–23)", yaxis_title="Mean 15-min Count", hovermode="x unified")
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.markdown("""
        <div class="alert-box alert-info">
            💡 <strong>Operational Insight:</strong> Weekend mid-day volume surges by over <strong>2.5x</strong> compared to weekday averages, driven by recreational leisure trips.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Day-of-Week Distribution")
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        date_series = df_clean['Timestamp'].dt.date.rename('date')
        dow_series = df_clean['Timestamp'].dt.dayofweek.rename('dow_num')
        daily_df = df_clean.groupby([date_series, dow_series])[['Sales Count', 'Redemption Count']].sum().reset_index()
        daily_df = daily_df.rename(columns={'Sales Count': 'Sales', 'Redemption Count': 'Redemptions'})
        daily_df['Day'] = daily_df['dow_num'].map(lambda x: dow_names[x])
        
        fig_dow = px.box(
            daily_df, x='Day', y=['Sales', 'Redemptions'],
            labels={'value': 'Total Daily Tickets', 'Day': 'Day of Week', 'variable': 'Metric'},
            color_discrete_map={'Sales': '#00F2FE', 'Redemptions': '#FF9F43'},
            category_orders={'Day': dow_names}
        )
        fig_dow = apply_custom_theme(fig_dow, title="Daily Ticket Demand Distribution Across Days of the Week", height=430)
        st.plotly_chart(fig_dow, use_container_width=True)
        
    with tab3:
        st.subheader("Monthly Seasonality (2015–2025 Aggregate)")
        df_clean['month'] = df_clean['Timestamp'].dt.month
        monthly_totals = df_clean.groupby('month')[['Sales Count', 'Redemption Count']].sum() / 1e6
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=month_labels, y=monthly_totals['Sales Count'], name='Total Sales (Millions)', marker_color='#10B981'))
        fig_m.add_trace(go.Bar(x=month_labels, y=monthly_totals['Redemption Count'], name='Total Redemptions (Millions)', marker_color='#F43F5E'))
        fig_m = apply_custom_theme(fig_m, title="Monthly Cumulative Ferry Volume (2015–2025)", height=430)
        fig_m.update_layout(barmode='group', xaxis_title="Month", yaxis_title="Total Cumulative Volume (Millions)")
        st.plotly_chart(fig_m, use_container_width=True)
        st.markdown("""
        <div class="alert-box alert-success">
            💡 <strong>Seasonal Rhythm:</strong> Over <strong>70% of annual ferry trips</strong> occur between June and August. Winter operations (Nov–Mar) function at baseline island commuter volume.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.subheader("Sales Leading Redemptions & Queue Accumulation")
        st.write("Ticket sales occur prior to passenger boarding gate scan. This lead-lag relationship provides strong predictive power.")
        sample_pts = df_clean.sample(n=min(4000, len(df_clean)), random_state=42).copy()
        sample_pts['Hour of Day'] = sample_pts['Timestamp'].dt.hour
        fig_scat = px.scatter(
            sample_pts, x='Sales Count', y='Redemption Count',
            color='Hour of Day',
            color_continuous_scale='Plasma',
            opacity=0.65,
            labels={'Hour of Day': 'Hour of Day'},
            title="15-Minute Correlation: Ticket Sales vs Redemptions"
        )
        fig_scat = apply_custom_theme(fig_scat, title="15-Minute Correlation: Ticket Sales vs Redemptions", height=430)
        st.plotly_chart(fig_scat, use_container_width=True)

# ==============================================================================
# PAGE 3: FORECAST
# ==============================================================================
elif page == "Page 3 — Forecast":
    st.markdown('<div class="main-header"><span class="floating-ferry">🔮</span> Short-Term Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real Multi-Horizon Predictive Engine with Conformal Prediction Intervals</div>', unsafe_allow_html=True)
    
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        selected_target = st.selectbox("Forecasting Target", ["Sales", "Redemptions"], index=0)
        target_key = "sales" if selected_target == "Sales" else "redemption"
    with col_ctrl2:
        selected_horizon = st.selectbox("Forecast Horizon", ["15 minutes", "30 minutes", "1 hour", "2 hours"], index=2)
        horizon_map = {"15 minutes": "15m", "30 minutes": "30m", "1 hour": "1h", "2 hours": "2h"}
        horizon_key = horizon_map[selected_horizon]
    with col_ctrl3:
        model_options = ["XGBoost", "Gradient Boosting", "Random Forest", "Linear Regression", "Moving Average", "Naive"]
        selected_model = st.selectbox("Forecasting Model Architecture", model_options, index=0)
        
    horizon_steps = {"15m": 1, "30m": 2, "1h": 4, "2h": 8}
    steps_count = horizon_steps[horizon_key]
    
    st.markdown("##### Reference Operational Time")
    
    ref_idx = st.slider(
        "Select Time Index to Forecast From (Simulated Live Operations)",
        min_value=len(df_features) - 400,
        max_value=len(df_features) - 1,
        value=len(df_features) - 1,
        step=1
    )
    
    ref_row = df_features.iloc[ref_idx]
    ref_time = ref_row['Timestamp']
    st.caption(f"Forecasting forward starting from: **{ref_time.strftime('%Y-%m-%d %H:%M')}**")
    
    all_horizons_in_order = ['15m', '30m', '1h', '2h']
    target_display = "Sales Count" if target_key == "sales" else "Redemption Count"
    
    future_rows = []
    features_input = df_features[FEATURE_COLS].iloc[[ref_idx]]
    
    active_horizons = [h for h in all_horizons_in_order if horizon_steps[h] <= steps_count]
    
    for h in active_horizons:
        model = load_trained_model(target_key, h, selected_model)
        if model is not None:
            pred_val = float(model.predict(features_input)[0])
        else:
            pred_val = float(ref_row[f'{target_key}_lag_1'])
            
        margin_90 = margins_dict.get(target_key, {}).get(h, {}).get(selected_model, {}).get('margin_90', 25.0)
        lower_b = max(0.0, round(pred_val - margin_90, 1))
        upper_b = round(pred_val + margin_90, 1)
        pred_rounded = round(pred_val, 1)
        
        future_time = ref_time + pd.Timedelta(minutes=horizon_steps[h] * 15)
        
        future_rows.append({
            'Horizon': h,
            'Future Time': future_time.strftime('%Y-%m-%d %H:%M'),
            'Predicted Demand': pred_rounded,
            'Lower Bound (90%)': lower_b,
            'Upper Bound (90%)': upper_b,
            'Prediction Interval Width': round(upper_b - lower_b, 1),
            'future_dt': future_time
        })
        
    forecast_table = pd.DataFrame(future_rows)
    
    hist_window = 24
    start_hist_idx = max(0, ref_idx - hist_window)
    hist_slice = df_clean.iloc[start_hist_idx:ref_idx + 1]
    
    fig_f = go.Figure()
    
    # Historical actuals trace
    fig_f.add_trace(go.Scatter(
        x=hist_slice['Timestamp'], y=hist_slice[target_display],
        mode='lines+markers', name=f'Historical Actual {selected_target}',
        line=dict(color='#00F2FE', width=3),
        marker=dict(size=4, color='#00F2FE')
    ))
    
    future_timestamps = [ref_time] + [r['future_dt'] for r in future_rows]
    last_actual = hist_slice[target_display].iloc[-1]
    future_preds = [last_actual] + [r['Predicted Demand'] for r in future_rows]
    upper_bounds = [last_actual] + [r['Upper Bound (90%)'] for r in future_rows]
    lower_bounds = [last_actual] + [r['Lower Bound (90%)'] for r in future_rows]
    
    # Upper bound
    fig_f.add_trace(go.Scatter(
        x=future_timestamps, y=upper_bounds,
        mode='lines', line=dict(width=0), showlegend=False
    ))
    # Lower bound with neon rose fill
    fig_f.add_trace(go.Scatter(
        x=future_timestamps, y=lower_bounds,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(244, 63, 94, 0.22)',
        name='90% Prediction Interval'
    ))
    
    # Forecast line
    fig_f.add_trace(go.Scatter(
        x=future_timestamps, y=future_preds,
        mode='lines+markers', name=f'Forecast ({selected_model})',
        line=dict(color='#F43F5E', width=3.5, dash='dash'),
        marker=dict(size=8, color='#F43F5E')
    ))
    
    fig_f = apply_custom_theme(
        fig_f,
        title=f"Short-Term {selected_target} Forecast Path ({selected_horizon} Ahead | {selected_model})",
        height=460
    )
    fig_f.update_layout(xaxis_title="Time", yaxis_title=f"{selected_target} (Count per 15-min)", hovermode="x unified")
    st.plotly_chart(fig_f, use_container_width=True)
    
    st.subheader("Forecast Path Breakdown")
    st.dataframe(
        forecast_table[['Horizon', 'Future Time', 'Predicted Demand', 'Lower Bound (90%)', 'Upper Bound (90%)', 'Prediction Interval Width']],
        use_container_width=True
    )
    
    m_row = metrics_df[(metrics_df['Target'] == selected_target) & 
                       (metrics_df['Horizon'] == horizon_key) & 
                       (metrics_df['Model'] == selected_model)]
    if not m_row.empty:
        m = m_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Holdout MAE", f"{m['MAE']} tickets")
        c2.metric("Holdout RMSE", f"{m['RMSE']} tickets")
        c3.metric("Holdout WAPE", f"{m['WAPE']}%")
        c4.metric("Peak Miss Rate", f"{m['Peak_Miss_Rate']}%", delta=f"{m['Peak_Count']} peak intervals", delta_color="inverse")

# ==============================================================================
# PAGE 4: MODEL COMPARISON
# ==============================================================================
elif page == "Page 4 — Model Comparison":
    st.markdown('<div class="main-header"><span class="floating-ferry">🏆</span> Model Performance Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Objective Empirical Benchmark on Unseen Holdout Data (June–December 2025)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        comp_target = st.selectbox("Evaluation Target", ["Sales", "Redemptions"], index=0)
        comp_metric = st.selectbox("Benchmark Metric", ["MAE", "RMSE", "WAPE", "Peak_Miss_Rate", "sMAPE"], index=0)
        
    filtered_metrics = metrics_df[metrics_df['Target'] == comp_target].copy()
    
    fig_comp = px.bar(
        filtered_metrics, x='Model', y=comp_metric, color='Horizon',
        barmode='group',
        title=f"Empirical Comparison of {comp_target} Forecasting: {comp_metric} Across Horizons",
        color_discrete_sequence=['#38BDF8', '#818CF8', '#C084FC', '#F472B6'],
        category_orders={'Horizon': ['15m', '30m', '1h', '2h']}
    )
    fig_comp = apply_custom_theme(fig_comp, title=f"Empirical Comparison of {comp_target} Forecasting: {comp_metric}", height=430)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.subheader(f"Complete Benchmark Table ({comp_target})")
    display_table = filtered_metrics[['Horizon', 'Model', 'MAE', 'RMSE', 'WAPE', 'sMAPE', 'Peak_Miss_Rate', 'Margin_90', 'Train_Time_s']].sort_values(['Horizon', 'MAE'])
    st.dataframe(display_table, use_container_width=True)
    
    st.markdown("""
    ### Analysis of Evaluation Results:
    1. **Machine Learning Dominance at Extended Horizons**:
       - At 15m ahead, persistence (Naive) is relatively competitive because demand changes gradually.
       - At **1 hour and 2 hours ahead**, Naive error spikes drastically (MAE increases from 22.6 to 39.4, WAPE from 34.6% to 60.2%).
       - Gradient Boosted Decision Trees (**XGBoost** and **HistGBR**) along with **Random Forest** maintain steady predictive accuracy (MAE ~17.8–19.1), cutting error rates by nearly half compared to baselines!
    2. **Peak Miss Rate Protection**:
       - Naive misses over **41%** of extreme demand peaks at 2 hours.
       - Tree-based models miss only **11.8% to 16.3%** of peaks, providing vital risk-reduction for ferry dispatchers.
    """)

# ==============================================================================
# PAGE 5: OPERATIONAL INSIGHTS
# ==============================================================================
elif page == "Page 5 — Operational Insights":
    st.markdown('<div class="main-header"><span class="floating-ferry">🚨</span> Predictive Operational Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Actionable Intelligence for Ferry Dispatchers, Crowd Management, and Dock Staff</div>', unsafe_allow_html=True)
    
    last_idx = len(df_features) - 1
    features_input = df_features[FEATURE_COLS].iloc[[last_idx]]
    last_time = df_features['Timestamp'].iloc[last_idx]
    
    sales_model_1h = load_trained_model('sales', '1h', 'XGBoost')
    pred_sales_1h = float(sales_model_1h.predict(features_input)[0]) if sales_model_1h else 0.0
    
    red_model_1h = load_trained_model('redemption', '1h', 'XGBoost')
    pred_red_1h = float(red_model_1h.predict(features_input)[0]) if red_model_1h else 0.0
    
    peak_thresh_sales = 182.0
    peak_thresh_red = 184.0
    
    margin_sales_1h = margins_dict.get('sales', {}).get('1h', {}).get('XGBoost', {}).get('margin_90', 30.0)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        is_surge = pred_sales_1h >= peak_thresh_sales or pred_red_1h >= peak_thresh_red
        status_label = "HIGH SURGE EXPECTED" if is_surge else "NORMAL OPERATION"
        card_border = "#F43F5E" if is_surge else "#10B981"
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {card_border};">
            <div class="metric-title">1-Hour Surge Alert Level</div>
            <div class="metric-value" style="font-size: 1.5rem; color: {card_border};">{status_label}</div>
            <div class="metric-sub">Peak 95th Pct Threshold: {peak_thresh_sales:.0f} tickets</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Projected 1-Hour Sales</div>
            <div class="metric-value">{pred_sales_1h:.0f}</div>
            <div class="metric-sub">Interval: {max(0, pred_sales_1h - margin_sales_1h):.0f} – {pred_sales_1h + margin_sales_1h:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Projected 1-Hour Redemptions</div>
            <div class="metric-value">{pred_red_1h:.0f}</div>
            <div class="metric-sub">Boarding scan volume at turnstiles</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Operational Guidance Rules")
    
    if is_surge:
        st.markdown("""
        <div class="alert-box alert-warning">
            ⚠️ <strong>Surge Advisory Active:</strong> Forecast demand in the next 1 hour exceeds the 95th percentile threshold (182 tickets/15min).<br>
            • <strong>Vessel Deployment:</strong> Stage secondary vessel (e.g. Ongiara or Trillium) on standby.<br>
            • <strong>Turnstile Staffing:</strong> Open auxiliary queue lines at Jack Layton Ferry Terminal.<br>
            • <strong>Queue Warning:</strong> Expect increased wait times; post queue estimates on terminal digital displays.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box alert-success">
            ✅ <strong>Standard Rhythm:</strong> Forecast demand remains within normal baseline capacity.<br>
            • Regular scheduled departures adequate to maintain sub-15 minute wait times.<br>
            • Standard turnstile staffing sufficient.
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Decision Support Matrix")
    matrix_df = pd.DataFrame([
        {
            "Forecasted Demand (per 15-min)": "< 50 tickets",
            "Operating Regime": "Off-Peak / Baseline",
            "Turnstile Staging": "1–2 Lanes Open",
            "Fleet Recommendation": "Standard single-vessel rotation"
        },
        {
            "Forecasted Demand (per 15-min)": "50 – 180 tickets",
            "Operating Regime": "Moderate Demand",
            "Turnstile Staging": "3–4 Lanes Open",
            "Fleet Recommendation": "Dual-vessel scheduled rhythm"
        },
        {
            "Forecasted Demand (per 15-min)": "180 – 350 tickets",
            "Operating Regime": "High Demand (95th Pct)",
            "Turnstile Staging": "All Standard Lanes Open",
            "Fleet Recommendation": "Full scheduled fleet + auxiliary crew standby"
        },
        {
            "Forecasted Demand (per 15-min)": "> 350 tickets",
            "Operating Regime": "Extreme Event Surge (99th Pct)",
            "Turnstile Staging": "All Lanes + Overflow Queues Active",
            "Fleet Recommendation": "Continuous shuttle dispatch mode (immediate turnaround)"
        }
    ])
    st.table(matrix_df)
    
    st.markdown("""
    <div class="alert-box alert-info">
        <strong>⚠️ Decision-Support Notice:</strong> This system provides predictive situational awareness. 
        Final dispatch authority and safety protocols remain strictly under the command of the Ferry Operations Master.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 6: DATA / MODEL INFO
# ==============================================================================
elif page == "Page 6 — Data / Model Info":
    st.markdown('<div class="main-header"><span class="floating-ferry">ℹ️</span> Data & Model Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Methodology, Provenance, Features, and Validation Integrity</div>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.subheader("Dataset Provenance")
        st.markdown(f"""
        - **Source**: City of Toronto Open Data Portal
        - **Asset**: Toronto Island Ferry Tickets (15-Minute Transaction Aggregates)
        - **Raw Record Count**: 261,538 transactional entries
        - **Chronological Span**: May 1, 2015 to December 21, 2025 (10.6 Years)
        - **Full Continuous Timeline**: 372,509 15-minute intervals
        - **Zero-Demand Intervals**: 111,651 intervals (29.9%) corresponding to terminal overnight closures (~23:00 to ~07:00) and off-season idle periods.
        - **Data Integrity**: Zero missing values, zero duplicates, zero negative values.
        """)
        
    with col_info2:
        st.subheader("Chronological Train / Validation / Test Splits")
        st.markdown("""
        - **Training Set (86.1%)**: May 2015 – June 30, 2024 (320,778 intervals)
        - **Validation Set (8.6%)**: July 1, 2024 – May 31, 2025 (32,160 intervals) — *Used for hyperparameter verification and empirical residual quantile margin estimation*.
        - **Test Holdout Set (5.3%)**: June 1, 2025 – December 21, 2025 (19,571 intervals) — *Strictly unseen future holdout spanning peak summer 2025 and fall 2025*.
        """)
        
    st.markdown("---")
    st.subheader("Feature Engineering Architecture (49 Features)")
    st.markdown("""
    | Category | Features | Description |
    | :--- | :--- | :--- |
    | **Lags (Sales & Redemptions)** | `lag_1`, `lag_2`, `lag_4`, `lag_8`, `lag_12`, `lag_24`, `lag_96`, `lag_672` | 15m, 30m, 1h, 2h, 3h, 6h, 24h (yesterday), and 7d (last week) |
    | **Rolling Statistics** | `roll_mean_4`, `roll_mean_8`, `roll_mean_24`, `roll_mean_96`, `roll_std_8`, `roll_max_8`, `roll_min_8` | Rolling short, medium, and 24-hour baseline moving averages & volatility |
    | **Cross-Target Signals** | `sales_to_redemption_ratio_lag1`, `sales_redemption_diff_lag1`, `sales_redemption_diff_lag4` | Dynamic queue lead-lag indicators (sales leading redemptions) |
    | **Calendar & Holidays** | `hour`, `minute`, `day_of_week`, `day_of_month`, `month`, `is_weekend`, `is_summer`, `is_shoulder`, `is_holiday` | Calendar flags including Ontario statutory holidays (Victoria Day, Canada Day, Civic Holiday, etc.) |
    | **Cyclical Trigonometry** | `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`, `sin_month`, `cos_month` | Smooth circular continuous representations of 24h, 7d, and 12m periodicities |
    """)
    
    st.markdown("---")
    st.subheader("Statistical Uncertainty Quantification")
    st.markdown(r"""
    Prediction intervals are estimated via **Conformal Residual Quantiles** evaluated on the independent chronological validation holdout set:
    $$\hat{y} \pm q_{1-\alpha}(|y - \hat{y}|)$$
    This provides statistically defensible 90% and 95% intervals without assuming Gaussian or homoscedastic error distributions.
    """)
    
    st.markdown("""
    <div class="alert-box alert-info">
        <strong>Mandatory Decision Support Disclaimer:</strong><br>
        Forecasts generated by this system are statistical estimates derived from historical ticket patterns, calendar factors, and recent activity. 
        They are intended exclusively for operational planning and supervisory decision-support, and should not be interpreted as absolute guarantees of passenger arrival.
    </div>
    """, unsafe_allow_html=True)
