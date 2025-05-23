import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import json
import os
import time

from utils.data_processor import load_sample_data
from utils.visualization import plot_rmse_over_time

# Set page title
st.set_page_config(page_title="Model Monitoring", page_icon="📈", layout="wide")

# Header
st.title("Model Monitoring Dashboard")
st.markdown("Monitor model performance, data drift, and system metrics.")

# Sidebar for time range selection
st.sidebar.header("Time Range")
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
)

# Initialize or load mock monitoring data
if 'monitoring_data' not in st.session_state:
    # Create some mock historical data
    np.random.seed(42)
    
    # Create dates for the past 90 days
    end_date = datetime.datetime.now()
    dates = [end_date - datetime.timedelta(days=i) for i in range(90)]
    dates.reverse()  # Oldest to newest
    
    # Generate mock metrics with some trends and variations
    rmse_values = []
    mae_values = []
    r2_values = []
    prediction_counts = []
    
    # Base values
    base_rmse = 3.5
    base_mae = 2.8
    base_r2 = 0.85
    
    # Add some drift and weekly patterns
    for i, date in enumerate(dates):
        # Add slight upward trend to error metrics (model degradation)
        drift = i * 0.005
        
        # Add weekly pattern (weekends have higher errors)
        weekly = 0.2 if date.weekday() >= 5 else 0
        
        # Add some random noise
        noise_rmse = np.random.normal(0, 0.2)
        noise_mae = np.random.normal(0, 0.15)
        noise_r2 = np.random.normal(0, 0.02)
        
        # Calculate metrics
        rmse = base_rmse + drift + weekly + noise_rmse
        mae = base_mae + drift * 0.8 + weekly * 0.7 + noise_mae
        r2 = max(0, min(1, base_r2 - drift * 0.1 - weekly * 0.05 + noise_r2))
        
        # Generate random prediction count with an upward trend (more usage over time)
        count = int(100 + i * 0.5 + np.random.normal(0, 10))
        
        # Append to lists
        rmse_values.append(rmse)
        mae_values.append(mae)
        r2_values.append(r2)
        prediction_counts.append(count)
    
    # Create DataFrame
    st.session_state.monitoring_data = pd.DataFrame({
        'timestamp': dates,
        'rmse': rmse_values,
        'mae': mae_values,
        'r2': r2_values,
        'prediction_count': prediction_counts
    })

# Get monitoring data
monitoring_data = st.session_state.monitoring_data

# Filter data based on selected time range
if time_range == "Last 7 Days":
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)
    filtered_data = monitoring_data[monitoring_data['timestamp'] >= cutoff_date]
elif time_range == "Last 30 Days":
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
    filtered_data = monitoring_data[monitoring_data['timestamp'] >= cutoff_date]
elif time_range == "Last 90 Days":
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=90)
    filtered_data = monitoring_data[monitoring_data['timestamp'] >= cutoff_date]
else:  # All Time
    filtered_data = monitoring_data

# Calculate current metrics (latest values)
current_metrics = {
    'rmse': filtered_data['rmse'].iloc[-1],
    'mae': filtered_data['mae'].iloc[-1],
    'r2': filtered_data['r2'].iloc[-1]
}

# Calculate changes from the first date in the filtered range
initial_metrics = {
    'rmse': filtered_data['rmse'].iloc[0],
    'mae': filtered_data['mae'].iloc[0],
    'r2': filtered_data['r2'].iloc[0]
}

metric_changes = {
    'rmse': current_metrics['rmse'] - initial_metrics['rmse'],
    'mae': current_metrics['mae'] - initial_metrics['mae'],
    'r2': current_metrics['r2'] - initial_metrics['r2']
}

# Create metric cards
st.header("Current Performance Metrics")

# Create columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="RMSE",
        value=f"{current_metrics['rmse']:.4f}",
        delta=f"{metric_changes['rmse']:.4f}",
        delta_color="inverse"  # Lower is better for error metrics
    )

with col2:
    st.metric(
        label="MAE",
        value=f"{current_metrics['mae']:.4f}",
        delta=f"{metric_changes['mae']:.4f}",
        delta_color="inverse"  # Lower is better for error metrics
    )

with col3:
    st.metric(
        label="R² Score",
        value=f"{current_metrics['r2']:.4f}",
        delta=f"{metric_changes['r2']:.4f}"
    )

# Trend charts
st.header("Performance Trends")

# Create two-row layout
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    # RMSE over time
    rmse_fig = px.line(filtered_data, x='timestamp', y='rmse', 
                       title="RMSE Over Time")
    rmse_fig.update_layout(height=300)
    st.plotly_chart(rmse_fig, use_container_width=True)

with row1_col2:
    # R² over time
    r2_fig = px.line(filtered_data, x='timestamp', y='r2', 
                     title="R² Score Over Time")
    r2_fig.update_layout(height=300)
    st.plotly_chart(r2_fig, use_container_width=True)

# Second row
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # MAE over time
    mae_fig = px.line(filtered_data, x='timestamp', y='mae', 
                      title="MAE Over Time")
    mae_fig.update_layout(height=300)
    st.plotly_chart(mae_fig, use_container_width=True)

with row2_col2:
    # Prediction count (usage)
    count_fig = px.bar(filtered_data, x='timestamp', y='prediction_count', 
                       title="Daily Prediction Count")
    count_fig.update_layout(height=300)
    st.plotly_chart(count_fig, use_container_width=True)

# Data Drift Detection
st.header("Data Drift Detection")

# Create mock data drift information
def generate_mock_drift_data():
    # Feature names based on turbofan dataset
    feature_names = [f"sensor_{i}" for i in range(1, 6)]
    
    drift_data = []
    
    for feature in feature_names:
        # Generate random drift score (0-1, higher means more drift)
        drift_score = np.random.beta(1.5, 5)  # Skewed toward lower values
        
        # Add alert status
        if drift_score > 0.7:
            status = "High Drift"
            alert = True
        elif drift_score > 0.4:
            status = "Moderate Drift"
            alert = False
        else:
            status = "No Significant Drift"
            alert = False
        
        drift_data.append({
            'feature': feature,
            'drift_score': drift_score,
            'status': status,
            'alert': alert
        })
    
    return pd.DataFrame(drift_data)

drift_data = generate_mock_drift_data()

# Display drift table with color coding
def color_drift(val):
    if val > 0.7:
        return 'background-color: #ffcccc'
    elif val > 0.4:
        return 'background-color: #ffffcc'
    else:
        return 'background-color: #ccffcc'

st.subheader("Feature Drift Analysis")
st.dataframe(drift_data.style.applymap(color_drift, subset=['drift_score']))

# Plot drift scores
drift_fig = px.bar(drift_data, x='feature', y='drift_score', 
                  color='drift_score',
                  color_continuous_scale=['green', 'yellow', 'red'],
                  title="Feature Drift Scores")
st.plotly_chart(drift_fig, use_container_width=True)

# Alerts Section
st.header("Alerts & Recommendations")

# Generate some mock alerts based on the data
alerts = []

# Check for consistent RMSE increase
rmse_trend = filtered_data['rmse'].iloc[-5:].mean() - filtered_data['rmse'].iloc[:5].mean()
if rmse_trend > 0.2:
    alerts.append({
        'level': 'High',
        'message': "RMSE has been consistently increasing. Consider retraining the model.",
        'recommendation': "Schedule model retraining with recent data."
    })

# Check for data drift alerts
if any(drift_data['alert']):
    drifting_features = drift_data[drift_data['alert']]['feature'].tolist()
    alerts.append({
        'level': 'High',
        'message': f"Significant data drift detected in features: {', '.join(drifting_features)}",
        'recommendation': "Investigate data sources for these features and consider collecting new training data."
    })

# Check for R² decrease
r2_trend = filtered_data['r2'].iloc[-5:].mean() - filtered_data['r2'].iloc[:5].mean()
if r2_trend < -0.05:
    alerts.append({
        'level': 'Medium',
        'message': "R² score has decreased over time, indicating reduced model explanatory power.",
        'recommendation': "Review feature importance and consider adding new features or removing noisy ones."
    })

# Mock maintenance recommendation based on recent predictions
if len(filtered_data) > 0:
    alerts.append({
        'level': 'Info',
        'message': "Based on recent predictions, units 3, 7, and 12 might need maintenance in the next 30 days.",
        'recommendation': "Schedule predictive maintenance for these units."
    })

# Display alerts
if alerts:
    for i, alert in enumerate(alerts):
        if alert['level'] == 'High':
            st.error(f"🚨 **{alert['level']}**: {alert['message']}")
        elif alert['level'] == 'Medium':
            st.warning(f"⚠️ **{alert['level']}**: {alert['message']}")
        else:
            st.info(f"ℹ️ **{alert['level']}**: {alert['message']}")
        
        st.markdown(f"**Recommendation**: {alert['recommendation']}")
        
        if i < len(alerts) - 1:
            st.markdown("---")
else:
    st.success("No alerts at this time. Model is performing within expected parameters.")

# System Health Section
st.header("System Health")

# Create mock system metrics
def generate_system_metrics():
    return {
        'cpu_usage': np.random.uniform(10, 40),
        'memory_usage': np.random.uniform(30, 70),
        'prediction_latency': np.random.uniform(50, 200),  # ms
        'availability': np.random.uniform(99.5, 100)
    }

system_metrics = generate_system_metrics()

# Create gauge charts for system metrics
fig = make_subplots(
    rows=1, cols=4,
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}, 
            {'type': 'indicator'}, {'type': 'indicator'}]]
)

# CPU Usage
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=system_metrics['cpu_usage'],
        title={'text': "CPU Usage (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "lightyellow"},
                {'range': [80, 100], 'color': "lightcoral"}
            ],
            'bar': {'color': "darkblue"}
        }
    ),
    row=1, col=1
)

# Memory Usage
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=system_metrics['memory_usage'],
        title={'text': "Memory Usage (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "lightyellow"},
                {'range': [80, 100], 'color': "lightcoral"}
            ],
            'bar': {'color': "darkblue"}
        }
    ),
    row=1, col=2
)

# Prediction Latency
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=system_metrics['prediction_latency'],
        title={'text': "Prediction Latency (ms)"},
        gauge={
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 100], 'color': "lightgreen"},
                {'range': [100, 250], 'color': "lightyellow"},
                {'range': [250, 500], 'color': "lightcoral"}
            ],
            'bar': {'color': "darkblue"}
        }
    ),
    row=1, col=3
)

# Availability
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=system_metrics['availability'],
        title={'text': "Service Availability (%)"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [95, 100]},
            'steps': [
                {'range': [95, 98], 'color': "lightyellow"},
                {'range': [98, 100], 'color': "lightgreen"}
            ],
            'bar': {'color': "darkblue"}
        }
    ),
    row=1, col=4
)

fig.update_layout(height=250)
st.plotly_chart(fig, use_container_width=True)

# Show last refresh time
st.sidebar.markdown("---")
st.sidebar.caption(f"Last dashboard refresh: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Refresh button
if st.sidebar.button("Refresh Dashboard"):
    st.experimental_rerun()
