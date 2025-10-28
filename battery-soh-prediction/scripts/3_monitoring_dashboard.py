# monitoring_dashboard.py - Streamlit monitoring app

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Battery SOH Monitor")

st.title("🔋 Battery SOH Real-Time Monitoring Dashboard")

# Simulate real-time data
data = load_real_time_predictions()  # From API or database

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_soh = data['soh'].mean()
    st.metric("Average SOH", f"{avg_soh:.1%}", 
             delta=f"{avg_soh-0.85:.1%}" if avg_soh > 0.85 else None)

with col2:
    critical_count = len(data[data['status'] == 'CRITICAL'])
    st.metric("Critical Batteries", critical_count, delta=f"-2 since yesterday")

with col3:
    avg_error = data['prediction_error'].mean()
    st.metric("Avg Prediction Error", f"±{avg_error:.2%}")

with col4:
    prediction_drift = check_drift(data)
    st.metric("Model Drift", "Nominal" if not prediction_drift else "⚠️ Alert")

# Time-series plot
st.subheader("SOH Trends Over Time")
fig = go.Figure()
for battery_id in data['battery_id'].unique()[:5]:
    battery_data = data[data['battery_id'] == battery_id].sort_values('timestamp')
    fig.add_trace(go.Scatter(
        x=battery_data['timestamp'],
        y=battery_data['soh'],
        mode='lines+markers',
        name=f'Battery {battery_id}',
        fill='tozeroy' if battery_id == 0 else None
    ))
st.plotly_chart(fig, use_container_width=True)

# Alert system
st.subheader("⚠️ Alerts & Warnings")
alerts = generate_alerts(data)
for alert in alerts:
    st.warning(alert)

# Detailed statistics
st.subheader("📊 Detailed Statistics")
summary_stats = data.groupby('status').agg({
    'soh': ['mean', 'std', 'count']
}).round(4)
st.dataframe(summary_stats)