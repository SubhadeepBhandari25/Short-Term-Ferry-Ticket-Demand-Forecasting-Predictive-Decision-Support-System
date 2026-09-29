"""
visualization.py
Visualization functions using Plotly and Matplotlib for ferry demand forecasting.
Generates publication-quality charts and interactive dashboard components.
"""

import os
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Theme styling for Matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_hourly_profiles_plotly(df: pd.DataFrame) -> go.Figure:
    """Generates an interactive diurnal hourly profile for Sales and Redemptions."""
    hourly = df.groupby(['hour', 'is_weekend'])[['Sales Count', 'Redemption Count']].mean().reset_index()
    hourly['Day Type'] = hourly['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    
    fig = go.Figure()
    
    # Weekday Sales
    wday = hourly[hourly['Day Type'] == 'Weekday']
    wend = hourly[hourly['Day Type'] == 'Weekend']
    
    fig.add_trace(go.Scatter(
        x=wday['hour'], y=wday['Sales Count'],
        mode='lines+markers', name='Sales (Weekday)',
        line=dict(color='#1f77b4', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=wend['hour'], y=wend['Sales Count'],
        mode='lines+markers', name='Sales (Weekend)',
        line=dict(color='#1f77b4', width=3, dash='dash')
    ))
    
    # Redemptions
    fig.add_trace(go.Scatter(
        x=wday['hour'], y=wday['Redemption Count'],
        mode='lines+markers', name='Redemption (Weekday)',
        line=dict(color='#ff7f0e', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=wend['hour'], y=wend['Redemption Count'],
        mode='lines+markers', name='Redemption (Weekend)',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Hourly Average Demand Profile (Weekday vs Weekend)",
        xaxis_title="Hour of Day (0–23)",
        yaxis_title="Average Ticket Count (15-min interval)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_dow_distribution_plotly(df: pd.DataFrame) -> go.Figure:
    """Generates day-of-week demand comparison."""
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby([df['Timestamp'].dt.date, 'day_of_week'])[['Sales Count', 'Redemption Count']].sum().reset_index()
    daily['Day'] = daily['day_of_week'].map(lambda x: dow_names[x])
    
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=daily['Day'], y=daily['Sales Count'],
        name='Daily Ticket Sales', marker_color='#1f77b4'
    ))
    fig.add_trace(go.Box(
        x=daily['Day'], y=daily['Redemption Count'],
        name='Daily Redemptions', marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="Daily Demand Distribution Across Days of the Week",
        xaxis_title="Day of Week",
        yaxis_title="Total Daily Tickets",
        boxmode='group',
        template="plotly_white"
    )
    return fig

def plot_sales_vs_redemption_scatter(df: pd.DataFrame, sample_size: int = 5000) -> go.Figure:
    """Scatter plot showing sales vs redemption relationship and queuing behavior."""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    fig = px.scatter(
        sample, x='Sales Count', y='Redemption Count',
        color='hour',
        color_continuous_scale='Viridis',
        opacity=0.6,
        title="15-Minute Ticket Sales vs Redemptions Correlation",
        labels={'Sales Count': 'Sales Count', 'Redemption Count': 'Redemption Count', 'hour': 'Hour'}
    )
    max_val = max(sample['Sales Count'].quantile(0.999), sample['Redemption Count'].quantile(0.999))
    fig.add_shape(
        type="line", line=dict(dash="dash", color="gray"),
        x0=0, y0=0, x1=max_val, y1=max_val
    )
    fig.update_layout(template="plotly_white")
    return fig

def plot_forecast_plotly(hist_timestamps: pd.Series, 
                         hist_actuals: pd.Series,
                         future_timestamps: pd.Series,
                         future_preds: pd.Series,
                         lower_bound: pd.Series,
                         upper_bound: pd.Series,
                         target_label: str = "Ticket Sales") -> go.Figure:
    """
    Renders the core interactive forecast chart with historical actuals,
    future predicted path, and shaded prediction interval band.
    """
    fig = go.Figure()
    
    # Historical actuals
    fig.add_trace(go.Scatter(
        x=hist_timestamps, y=hist_actuals,
        mode='lines', name=f'Historical {target_label}',
        line=dict(color='#2b5c8f', width=2.5)
    ))
    
    # Upper bound of prediction interval
    fig.add_trace(go.Scatter(
        x=future_timestamps, y=upper_bound,
        mode='lines', line=dict(width=0),
        showlegend=False, name='Upper Bound'
    ))
    
    # Lower bound with fill to upper
    fig.add_trace(go.Scatter(
        x=future_timestamps, y=lower_bound,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(231, 76, 60, 0.22)',
        name='Prediction Interval (90%)'
    ))
    
    # Future prediction
    fig.add_trace(go.Scatter(
        x=future_timestamps, y=future_preds,
        mode='lines+markers', name=f'Forecasted {target_label}',
        line=dict(color='#e74c3c', width=3, dash='dash'),
        marker=dict(size=7, color='#e74c3c')
    ))
    
    fig.update_layout(
        title=f"Short-Term {target_label} Forecast & Uncertainty Band",
        xaxis_title="Time",
        yaxis_title=f"{target_label} (per 15-min interval)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_model_comparison_bar(metrics_df: pd.DataFrame, metric_col: str = 'MAE') -> go.Figure:
    """Generates comparison bar chart across models and horizons."""
    fig = px.bar(
        metrics_df, x='Model', y=metric_col, color='Horizon',
        barmode='group',
        title=f"Model Performance Comparison: {metric_col} by Horizon",
        labels={'Model': 'Forecasting Model', metric_col: metric_col, 'Horizon': 'Forecast Horizon'},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(template="plotly_white")
    return fig

def export_static_eda_figures(df: pd.DataFrame, output_dir: str):
    """Exports high-resolution static PNG figures for reports and paper documentation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Diurnal Hourly Trend
    hourly = df.groupby('hour')[['Sales Count', 'Redemption Count']].mean()
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(hourly.index, hourly['Sales Count'], marker='o', label='Mean Ticket Sales', color='#1f77b4', lw=2)
    plt.plot(hourly.index, hourly['Redemption Count'], marker='s', label='Mean Redemptions', color='#ff7f0e', lw=2)
    plt.title('Toronto Island Ferry: Hourly Demand Profile (2015–2025)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Count per 15-min')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_profile.png'))
    plt.close()
    
    # 2. Monthly Seasonality
    monthly = df.groupby('month')[['Sales Count', 'Redemption Count']].sum() / 1e6
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure(figsize=(10, 5), dpi=300)
    bar_width = 0.35
    x = np.arange(len(months))
    plt.bar(x - bar_width/2, monthly['Sales Count'], width=bar_width, label='Sales (Millions)', color='#2ca02c')
    plt.bar(x + bar_width/2, monthly['Redemption Count'], width=bar_width, label='Redemptions (Millions)', color='#d62728')
    plt.title('Monthly Seasonal Ferry Volume (Cumulative 2015–2025)')
    plt.xlabel('Month')
    plt.ylabel('Total Tickets (Millions)')
    plt.xticks(x, months)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_seasonality.png'))
    plt.close()
    
    # 3. Day of Week
    dow = df.groupby('day_of_week')[['Sales Count', 'Redemption Count']].mean()
    d_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.figure(figsize=(9, 5), dpi=300)
    plt.bar(d_names, dow['Sales Count'], label='Avg Sales', color='#3498db', alpha=0.85)
    plt.bar(d_names, dow['Redemption Count'], label='Avg Redemptions', color='#e67e22', alpha=0.7)
    plt.title('Average 15-min Volume by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'day_of_week_demand.png'))
    plt.close()
