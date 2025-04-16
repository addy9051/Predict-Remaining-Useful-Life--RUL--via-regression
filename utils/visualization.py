import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_sensor_data(df, unit_id=None, sensors=None):
    """
    Plot sensor data over time for a specific unit
    
    Args:
        df: DataFrame containing sensor data
        unit_id: Unit ID to plot (None plots all units)
        sensors: List of sensor columns to plot (None plots all sensors)
        
    Returns:
        Plotly figure
    """
    if 'unit_number' not in df.columns or 'time_cycles' not in df.columns:
        print("DataFrame must contain 'unit_number' and 'time_cycles' columns")
        return None
    
    # Filter for specific unit if provided
    if unit_id is not None:
        data = df[df['unit_number'] == unit_id].copy()
    else:
        data = df.copy()
    
    # Identify sensor columns if not provided
    if sensors is None:
        sensors = [col for col in df.columns if 'sensor' in col and not any(x in col for x in ['rolling', 'diff'])]
    
    # Create figure
    fig = make_subplots(rows=len(sensors), cols=1, 
                        shared_xaxes=True, 
                        subplot_titles=sensors,
                        vertical_spacing=0.02)
    
    # Color by unit if multiple units
    if unit_id is None:
        for i, sensor in enumerate(sensors, 1):
            fig.add_trace(
                go.Scatter(x=data['time_cycles'], y=data[sensor], 
                           mode='lines', 
                           customdata=data['unit_number'],
                           hovertemplate='Unit: %{customdata}<br>Cycle: %{x}<br>Value: %{y}',
                           line=dict(width=1),
                           opacity=0.7,
                           name=sensor),
                row=i, col=1
            )
    else:
        for i, sensor in enumerate(sensors, 1):
            fig.add_trace(
                go.Scatter(x=data['time_cycles'], y=data[sensor], 
                           mode='lines', 
                           name=sensor,
                           line=dict(width=2)),
                row=i, col=1
            )
    
    # Update layout
    height = max(100 * len(sensors), 400)
    fig.update_layout(height=height, width=800, 
                      title_text=f"Sensor Readings over Time{' for Unit '+str(unit_id) if unit_id else ''}",
                      showlegend=False)
    
    fig.update_xaxes(title_text="Cycles", row=len(sensors), col=1)
    
    return fig

def plot_rul_vs_cycles(df, unit_id=None):
    """
    Plot RUL vs Cycles
    
    Args:
        df: DataFrame containing RUL data
        unit_id: Unit ID to plot (None plots all units)
        
    Returns:
        Plotly figure
    """
    if 'RUL' not in df.columns or 'time_cycles' not in df.columns:
        print("DataFrame must contain 'RUL' and 'time_cycles' columns")
        return None
    
    # Filter for specific unit if provided
    if unit_id is not None:
        data = df[df['unit_number'] == unit_id].copy()
        title = f"RUL vs Cycles for Unit {unit_id}"
    else:
        data = df.copy()
        title = "RUL vs Cycles for All Units"
    
    # Create figure
    fig = px.line(data, x='time_cycles', y='RUL', color='unit_number' if unit_id is None else None,
                 title=title)
    
    fig.update_layout(height=500, width=800)
    fig.update_xaxes(title_text="Cycles")
    fig.update_yaxes(title_text="Remaining Useful Life (RUL)")
    
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for a trained model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        Plotly figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Take top 20 features for visibility
    if len(sorted_feature_names) > 20:
        sorted_feature_names = sorted_feature_names[:20]
        sorted_importances = sorted_importances[:20]
    
    # Create figure
    fig = px.bar(x=sorted_importances, y=sorted_feature_names, orientation='h',
                title="Feature Importance")
    
    fig.update_layout(height=max(300, len(sorted_feature_names) * 20), width=800)
    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="Feature")
    
    return fig

def plot_prediction_vs_actual(y_pred, y_true, title="Prediction vs Actual"):
    """
    Plot predicted vs actual values
    
    Args:
        y_pred: Predicted values
        y_true: Actual values
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Create DataFrame for plotting
    data = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Create scatter plot
    fig = px.scatter(data, x='Actual', y='Predicted',
                    title=title)
    
    # Add perfect prediction line
    max_val = max(data['Actual'].max(), data['Predicted'].max())
    min_val = min(data['Actual'].min(), data['Predicted'].min())
    
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', line=dict(dash='dash', color='red'),
                  name='Perfect Prediction')
    )
    
    fig.update_layout(height=600, width=800)
    fig.update_xaxes(title_text="Actual RUL")
    fig.update_yaxes(title_text="Predicted RUL")
    
    return fig

def plot_rmse_over_time(results_df):
    """
    Plot RMSE over time for model monitoring
    
    Args:
        results_df: DataFrame with columns 'timestamp' and 'rmse'
        
    Returns:
        Plotly figure
    """
    fig = px.line(results_df, x='timestamp', y='rmse',
                 title="RMSE Over Time")
    
    fig.update_layout(height=500, width=800)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="RMSE")
    
    return fig

def plot_error_distribution(y_pred, y_true):
    """
    Plot distribution of prediction errors
    
    Args:
        y_pred: Predicted values
        y_true: Actual values
        
    Returns:
        Plotly figure
    """
    # Calculate errors
    errors = y_pred - y_true
    
    # Create histogram
    fig = px.histogram(errors, nbins=30,
                      title="Distribution of Prediction Errors")
    
    fig.update_layout(height=500, width=800)
    fig.update_xaxes(title_text="Prediction Error")
    fig.update_yaxes(title_text="Frequency")
    
    return fig
