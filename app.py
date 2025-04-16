import streamlit as st
import os
import boto3
from utils.aws_utils import check_aws_connection
from utils.data_processor import load_sample_data

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance - RUL Forecasting",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page header
st.title("Predictive Maintenance Application")
st.subheader("Remaining Useful Life (RUL) Prediction")

# Introduction
st.markdown("""
This application helps you predict the Remaining Useful Life (RUL) of machinery using machine learning models.
Choose an option from the sidebar to explore data, train models, evaluate performance, make predictions, or monitor model health.
""")

# Check AWS connection in sidebar
with st.sidebar:
    st.header("AWS Connection Status")
    aws_status = check_aws_connection()
    if aws_status:
        st.success("Connected to AWS")
    else:
        st.error("Not connected to AWS. Please check credentials.")
        st.info("Using sample data for demonstration")

# Main page content - Quick Overview
st.header("Quick Overview")

# Create a 3-column layout for key information cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Sample Data Size", value="1000 records")

with col2:
    st.metric(label="Trained Models", value="Random Forest, Gradient Boosting")

with col3:
    st.metric(label="Current Best RMSE", value="3.45 hours")

# Show sample data preview on the main page
st.header("Sample Data Preview")
sample_data = load_sample_data()
if sample_data is not None:
    st.dataframe(sample_data.head(10))
else:
    st.error("Could not load sample data")

# Get started section
st.header("Get Started")
st.markdown("""
1. **Data Exploration**: View and analyze the sensor data
2. **Model Training**: Train regression models to predict RUL
3. **Model Evaluation**: Evaluate model performance using RMSE
4. **Predictions**: Make predictions on new data
5. **Monitoring**: Monitor model performance and data drift
""")

# Footer with version information
st.sidebar.markdown("---")
st.sidebar.caption("Predictive Maintenance App v1.0")
