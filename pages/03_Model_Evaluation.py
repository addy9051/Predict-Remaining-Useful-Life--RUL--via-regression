import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import time

from utils.visualization import (
    plot_prediction_vs_actual,
    plot_error_distribution,
    plot_feature_importance
)

# Set page title
st.set_page_config(page_title="Model Evaluation", page_icon="📊")

# Header
st.title("Model Evaluation")
st.markdown("Evaluate and analyze model performance for RUL prediction.")

# Check if models are available
if ('rf_model' not in st.session_state or st.session_state.rf_model is None) and \
   ('gb_model' not in st.session_state or st.session_state.gb_model is None) and \
   ('best_model' not in st.session_state or st.session_state.best_model is None):
    st.warning("No trained models available. Please train models in the Model Training page first.")
    st.stop()

# If no test data available
if 'X_test' not in st.session_state or st.session_state.X_test is None or \
   'y_test' not in st.session_state or st.session_state.y_test is None:
    st.warning("No test data available. Please preprocess data in the Model Training page first.")
    st.stop()

# Model selection
models = []
if 'rf_model' in st.session_state and st.session_state.rf_model is not None:
    models.append('Random Forest')
if 'gb_model' in st.session_state and st.session_state.gb_model is not None:
    models.append('Gradient Boosting')
if 'best_model' in st.session_state and st.session_state.best_model is not None and 'best_model_name' in st.session_state:
    models.append(f'Best Model ({st.session_state.best_model_name.replace("_", " ").title()})')

selected_model_name = st.selectbox("Select Model to Evaluate", models)

# Get the selected model
if selected_model_name == 'Random Forest':
    model = st.session_state.rf_model
elif selected_model_name == 'Gradient Boosting':
    model = st.session_state.gb_model
else:  # Best model
    model = st.session_state.best_model

# Get test data
X_test = st.session_state.X_test
y_test = st.session_state.y_test

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
st.header("Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RMSE", f"{rmse:.4f}")
    
with col2:
    st.metric("MAE", f"{mae:.4f}")
    
with col3:
    st.metric("R² Score", f"{r2:.4f}")

# Visualizations
st.header("Performance Visualizations")

# Predicted vs Actual
st.subheader("Predicted vs. Actual Values")
pred_vs_actual_fig = plot_prediction_vs_actual(y_pred, y_test, title=f"{selected_model_name}: Predicted vs. Actual RUL")
st.plotly_chart(pred_vs_actual_fig, use_container_width=True)

# Error distribution
st.subheader("Error Distribution")
error_dist_fig = plot_error_distribution(y_pred, y_test)
st.plotly_chart(error_dist_fig, use_container_width=True)

# Feature importance
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importance")
    
    # Get feature names
    feature_names = st.session_state.feature_names
    
    feature_imp_fig = plot_feature_importance(model, feature_names)
    st.plotly_chart(feature_imp_fig, use_container_width=True)

# Error Analysis
st.header("Error Analysis")

# Calculate absolute errors
errors = np.abs(y_pred - y_test)

# Find worst predictions
n_worst = st.slider("Number of worst predictions to display", min_value=5, max_value=50, value=10)
worst_indices = np.argsort(errors)[-n_worst:][::-1]

# Create DataFrame with worst predictions
worst_df = pd.DataFrame({
    'Actual RUL': y_test[worst_indices],
    'Predicted RUL': y_pred[worst_indices],
    'Absolute Error': errors[worst_indices]
})

st.write(f"Top {n_worst} worst predictions:")
st.dataframe(worst_df)

# Error distribution by RUL range
st.subheader("Error by RUL Range")

# Create RUL bins
bins = [0, 50, 100, 150, 200, np.inf]
labels = ['0-50', '51-100', '101-150', '151-200', '200+']

# Group errors by RUL range
error_by_range = pd.DataFrame({
    'Actual RUL': y_test,
    'Absolute Error': errors
})
error_by_range['RUL Range'] = pd.cut(error_by_range['Actual RUL'], bins=bins, labels=labels)

# Calculate average error by range
error_summary = error_by_range.groupby('RUL Range').agg({
    'Absolute Error': ['mean', 'std', 'count']
}).reset_index()
error_summary.columns = ['RUL Range', 'Mean Error', 'Std Error', 'Count']

# Display error summary
st.write("Error by RUL Range:")
st.dataframe(error_summary)

# Plot error by range
fig = px.bar(error_summary, x='RUL Range', y='Mean Error', 
             error_y='Std Error',
             labels={'Mean Error': 'Mean Absolute Error',
                    'RUL Range': 'RUL Range (hours)'},
             title="Mean Absolute Error by RUL Range")
st.plotly_chart(fig, use_container_width=True)

# Model interpretation (simple)
st.header("Simple Model Interpretation")

if hasattr(model, 'feature_importances_'):
    # Get top 5 features
    top_features = np.argsort(model.feature_importances_)[-5:]
    top_feature_names = [feature_names[i] for i in top_features]
    top_feature_importance = model.feature_importances_[top_features]
    
    # Create interpretation text
    st.write("Key factors influencing RUL prediction:")
    for name, importance in zip(top_feature_names, top_feature_importance):
        st.write(f"- **{name}**: {importance:.4f} importance weight")
    
    st.info("""
    **Interpretation Tips:**
    - Higher feature importance indicates stronger influence on predictions
    - Features with high importance should be monitored closely
    - Consider engineering new features from high-importance sensors
    """)
