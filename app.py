import streamlit as st
import os
import boto3
import pandas as pd
from utils.aws_utils import check_aws_connection
from utils.data_processor import load_sample_data, load_data_from_api, load_data_from_s3

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

# Data Source and Connection Status in sidebar
with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Select Data Source", ["NASA CMAPSS Data", "Sample Data", "API", "AWS S3"])
    
    if data_source == "NASA CMAPSS Data":
        st.subheader("NASA CMAPSS Dataset")
        # Initialize session state for NASA dataset settings
        if 'nasa_dataset' not in st.session_state:
            st.session_state.nasa_dataset = "FD001"
            
        # NASA dataset selection
        st.session_state.nasa_dataset = st.selectbox(
            "Select Dataset", 
            ["FD001", "FD002", "FD003", "FD004"],
            index=["FD001", "FD002", "FD003", "FD004"].index(st.session_state.nasa_dataset),
            help="FD001: Sea Level, Single Fault Mode; FD002: Six Operating Conditions, Single Fault Mode; " +
                 "FD003: Sea Level, Two Fault Modes; FD004: Six Operating Conditions, Two Fault Modes"
        )
        
        # Load NASA data button
        nasa_data_button = st.button("Load NASA Dataset")
        if nasa_data_button:
            with st.spinner(f"Loading NASA CMAPSS {st.session_state.nasa_dataset} dataset..."):
                from utils.data_processor import load_nasa_cmapss_data
                
                st.session_state.fetched_data = load_nasa_cmapss_data(
                    dataset=st.session_state.nasa_dataset
                )
                if st.session_state.fetched_data is not None:
                    st.success(f"Successfully loaded NASA dataset ({len(st.session_state.fetched_data)} records)")
                else:
                    st.error(f"Failed to load NASA dataset {st.session_state.nasa_dataset}.")
                    
        # NASA data information
        nasa_info = st.expander("Dataset Information")
        with nasa_info:
            st.markdown("""
            ### NASA CMAPSS Datasets
            
            - **FD001**: 100 engines, Sea Level conditions, Single fault mode (HPC Degradation)
            - **FD002**: 260 engines, Six operating conditions, Single fault mode (HPC Degradation)
            - **FD003**: 100 engines, Sea Level conditions, Two fault modes (HPC and Fan Degradation)
            - **FD004**: 248 engines, Six operating conditions, Two fault modes (HPC and Fan Degradation)
            
            Each dataset includes:
            - Training data (engines run to failure)
            - Test data (engines run up to a point before failure)
            - True RUL values for test data
            
            Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver CO, Oct 2008.
            """)
        
    elif data_source == "AWS S3":
        st.subheader("AWS Connection")
        aws_status = check_aws_connection()
        if aws_status:
            st.success("Connected to AWS")
            # Allow selecting S3 bucket and file
            if 'aws_bucket' not in st.session_state:
                st.session_state.aws_bucket = ""
            if 'aws_file_key' not in st.session_state:
                st.session_state.aws_file_key = ""
                
            st.session_state.aws_bucket = st.text_input("S3 Bucket Name", value=st.session_state.aws_bucket)
            st.session_state.aws_file_key = st.text_input("S3 File Key/Path", value=st.session_state.aws_file_key)
        else:
            st.error("Not connected to AWS. Please check credentials.")
            st.info("Defaulting to NASA data")
            data_source = "NASA CMAPSS Data"
    
    elif data_source == "API":
        st.subheader("API Configuration")
        # Initialize session state for API settings
        if 'api_url' not in st.session_state:
            st.session_state.api_url = ""
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        if 'dataset_type' not in st.session_state:
            st.session_state.dataset_type = "turbofan"
            
        # API configuration form
        st.session_state.api_url = st.text_input("API Endpoint URL", value=st.session_state.api_url, 
                               placeholder="https://api.example.com/data")
        st.session_state.api_key = st.text_input("API Key (optional)", value=st.session_state.api_key, 
                               placeholder="Enter your API key", type="password")
        st.session_state.dataset_type = st.selectbox("Dataset Type", 
                                    ["turbofan", "sensor", "custom"],
                                    index=["turbofan", "sensor", "custom"].index(st.session_state.dataset_type))
        
        api_info = st.expander("API Info")
        with api_info:
            st.markdown("""
            - **turbofan**: NASA Turbofan Engine Degradation Data format
            - **sensor**: General sensor data with timestamps
            - **custom**: Other data formats (will attempt to auto-detect structure)
            """)
    
    # Add "Fetch Data" button for API and S3 sources
    if data_source in ["API", "AWS S3"]:
        fetch_button = st.button("Fetch Data")
        if fetch_button:
            with st.spinner("Fetching data..."):
                if data_source == "API" and st.session_state.api_url:
                    st.session_state.fetched_data = load_data_from_api(
                        st.session_state.api_url, 
                        st.session_state.api_key if st.session_state.api_key else None,
                        st.session_state.dataset_type
                    )
                    if st.session_state.fetched_data is not None:
                        st.success(f"Successfully fetched data from API ({len(st.session_state.fetched_data)} records)")
                    else:
                        st.error("Failed to fetch data from API. Check URL and credentials.")
                
                elif data_source == "AWS S3" and st.session_state.aws_bucket and st.session_state.aws_file_key:
                    st.session_state.fetched_data = load_data_from_s3(
                        st.session_state.aws_bucket,
                        st.session_state.aws_file_key
                    )
                    if st.session_state.fetched_data is not None:
                        st.success(f"Successfully fetched data from S3 ({len(st.session_state.fetched_data)} records)")
                    else:
                        st.error("Failed to fetch data from S3. Check bucket and file path.")
                        
    # Save data source to session state
    st.session_state.data_source = data_source

# Main page content - Quick Overview
st.header("Quick Overview")

# Create a 3-column layout for key information cards
col1, col2, col3 = st.columns(3)

with col1:
    # Data size metric
    data_size_value = "N/A"
    if 'data' in st.session_state and st.session_state.data is not None:
        data_size_value = f"{len(st.session_state.data):,} records"
    
    data_source_label = "Data Size"
    if 'data_source' in st.session_state:
        data_source_label = f"{st.session_state.data_source} Size"
        
    st.metric(label=data_source_label, value=data_size_value)

with col2:
    # Count of available models
    trained_models = []
    if 'rf_model' in st.session_state and st.session_state.rf_model is not None:
        trained_models.append("Random Forest")
    if 'gb_model' in st.session_state and st.session_state.gb_model is not None:
        trained_models.append("Gradient Boosting")
    
    model_value = ", ".join(trained_models) if trained_models else "None"
    st.metric(label="Trained Models", value=model_value)

with col3:
    # Best model metric if available
    rmse_value = "N/A"
    if 'best_model_name' in st.session_state and st.session_state.best_model_name and 'train_metrics' in st.session_state:
        metrics = st.session_state.train_metrics.get(st.session_state.best_model_name, {})
        if 'test_rmse' in metrics:
            rmse_value = f"{metrics['test_rmse']:.2f} hours"
    
    st.metric(label="Current Best RMSE", value=rmse_value)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Get data based on selected source
if 'data_source' in st.session_state:
    data_source = st.session_state.data_source
    
    # Load data based on the selected source
    if data_source == "Sample Data":
        st.session_state.data = load_sample_data()
    elif data_source == "API" and 'fetched_data' in st.session_state and st.session_state.fetched_data is not None:
        st.session_state.data = st.session_state.fetched_data
    elif data_source == "AWS S3" and 'fetched_data' in st.session_state and st.session_state.fetched_data is not None:
        st.session_state.data = st.session_state.fetched_data
    else:
        # Default to sample data if nothing else is available
        st.session_state.data = load_sample_data()

# Show data preview
st.header("Data Preview")
if st.session_state.data is not None:
    # Show data source info
    source_info = ""
    if 'data_source' in st.session_state:
        if st.session_state.data_source == "Sample Data":
            source_info = "Using sample synthetic data for demonstration"
        elif st.session_state.data_source == "API" and 'api_url' in st.session_state:
            source_info = f"Data from API: {st.session_state.api_url}"
        elif st.session_state.data_source == "AWS S3" and 'aws_bucket' in st.session_state and 'aws_file_key' in st.session_state:
            source_info = f"Data from S3: {st.session_state.aws_bucket}/{st.session_state.aws_file_key}"
    
    st.info(source_info)
    
    # Data info
    data_info = st.expander("Data Information")
    with data_info:
        data_shape = st.session_state.data.shape
        st.write(f"Records: {data_shape[0]}, Features: {data_shape[1]}")
        
        # Check for essential columns
        required_cols = ['unit_number', 'time_cycles', 'RUL']
        missing_cols = [col for col in required_cols if col not in st.session_state.data.columns]
        if missing_cols:
            st.warning(f"Missing essential columns: {', '.join(missing_cols)}")
        
        # Show column types
        st.write("Column types:")
        st.write(st.session_state.data.dtypes)
        
        # Show basic statistics
        if 'RUL' in st.session_state.data.columns:
            st.write("RUL statistics:")
            st.write(st.session_state.data['RUL'].describe())
    
    # Show data preview
    st.dataframe(st.session_state.data.head(10))
else:
    st.error("No data available. Please check data source configuration.")

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
