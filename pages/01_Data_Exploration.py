import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processor import load_sample_data, load_data_from_s3, load_data_from_api
from utils.aws_utils import list_s3_buckets, list_s3_objects
from utils.visualization import plot_sensor_data, plot_rul_vs_cycles

# Set page title
st.set_page_config(page_title="Data Exploration", page_icon="📊")

# Header
st.title("Data Exploration")
st.markdown("Explore and analyze the machine sensor data for predictive maintenance.")

# Check if data is already in session state from main app
use_main_app_data = False
if 'data' in st.session_state and st.session_state.data is not None and 'data_source' in st.session_state:
    use_main_app_data = st.sidebar.checkbox("Use data from main app", value=True)

if use_main_app_data:
    # Use the data that was loaded in the main app
    data = st.session_state.data
    st.sidebar.info(f"Using {st.session_state.data_source} from main app")
    
    # Show which API URL the data came from if applicable
    if st.session_state.data_source == "API" and 'api_url' in st.session_state:
        st.sidebar.write(f"API Endpoint: {st.session_state.api_url}")
    elif st.session_state.data_source == "AWS S3" and 'aws_bucket' in st.session_state and 'aws_file_key' in st.session_state:
        st.sidebar.write(f"S3 Path: {st.session_state.aws_bucket}/{st.session_state.aws_file_key}")
        
else:
    # Sidebar for data source selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Sample Data", "API", "AWS S3 Bucket"]
    )
    
    # Initialize session state for data
    if 'exploration_data' not in st.session_state:
        st.session_state.exploration_data = None
    
    # Load data based on selection
    if data_source == "Sample Data":
        # Load sample data
        if st.session_state.exploration_data is None:
            st.session_state.exploration_data = load_sample_data()
        
        data = st.session_state.exploration_data
        
        if data is not None:
            st.sidebar.success("Sample data loaded successfully!")
        else:
            st.error("Failed to load sample data.")
            st.stop()
    
    elif data_source == "API":
        # API configuration
        st.sidebar.subheader("API Configuration")
        
        # API URL input
        api_url = st.sidebar.text_input("API Endpoint URL", placeholder="https://api.example.com/data")
        api_key = st.sidebar.text_input("API Key (optional)", type="password", placeholder="Enter API key if required")
        
        # Dataset type selection
        dataset_type = st.sidebar.selectbox(
            "Data Format",
            ["turbofan", "sensor", "custom"],
            help="Select the format of data returned by the API"
        )
        
        # Load data button
        if api_url and st.sidebar.button("Fetch Data from API"):
            with st.spinner("Fetching data from API..."):
                data = load_data_from_api(api_url, api_key, dataset_type)
                
                if data is not None:
                    st.session_state.exploration_data = data
                    st.sidebar.success(f"Data fetched from API: {len(data)} records")
                else:
                    st.error("Failed to fetch data from API. Check URL and credentials.")
                    st.stop()
        else:
            # Use existing data or sample data if no API data has been loaded
            data = st.session_state.exploration_data
            if data is None:
                st.sidebar.warning("No API data loaded yet. Enter an API URL and fetch data, or select another data source.")
                data = load_sample_data()
                st.sidebar.info("Using sample data for now")
    
    else:  # AWS S3 Bucket
        # Get available S3 buckets
        buckets = list_s3_buckets()
        
        if buckets:
            selected_bucket = st.sidebar.selectbox("Select S3 Bucket", buckets)
            
            # List objects in selected bucket
            objects = list_s3_objects(selected_bucket)
            
            if objects:
                selected_object = st.sidebar.selectbox("Select File", objects)
                
                # Load data button
                if st.sidebar.button("Load Data from S3"):
                    with st.spinner("Loading data from S3..."):
                        data = load_data_from_s3(selected_bucket, selected_object)
                        
                        if data is not None:
                            st.session_state.exploration_data = data
                            st.sidebar.success(f"Data loaded from S3: {selected_object}")
                        else:
                            st.error("Failed to load data from S3.")
                            st.stop()
                else:
                    # Use existing data or default to sample data
                    data = st.session_state.exploration_data
                    if data is None:
                        st.sidebar.warning("No S3 data loaded yet. Select a file and load it, or select another data source.")
                        data = load_sample_data()
                        st.sidebar.info("Using sample data for now")
            else:
                st.sidebar.warning("No objects found in the selected bucket.")
                data = load_sample_data()
        else:
            st.sidebar.warning("No S3 buckets available. Check your AWS credentials.")
            data = load_sample_data()

# Display data overview
st.header("Data Overview")

# Basic statistics
st.subheader("Summary Statistics")
st.write(data.describe())

# Display data shape
st.write(f"Data Shape: {data.shape[0]} rows, {data.shape[1]} columns")

# Display column info
st.subheader("Column Information")
column_info = pd.DataFrame({
    'Column Name': data.columns,
    'Data Type': data.dtypes,
    'Non-Null Count': data.count(),
    'Null Count': data.isnull().sum()
})
st.dataframe(column_info)

# Data preview
st.subheader("Data Preview")
st.dataframe(data.head(10))

# Visualization section
st.header("Data Visualization")

# Filter by unit
if 'unit_number' in data.columns:
    unique_units = sorted(data['unit_number'].unique())
    selected_unit = st.selectbox("Select Unit ID for Analysis", 
                                 options=[None] + list(unique_units),
                                 format_func=lambda x: "All Units" if x is None else f"Unit {x}")
else:
    selected_unit = None

# Sensor visualization
st.subheader("Sensor Data over Time")

# Select sensors to visualize
if 'sensor_1' in data.columns:
    sensor_cols = [col for col in data.columns if 'sensor' in col and not any(x in col for x in ['rolling', 'diff'])]
    selected_sensors = st.multiselect("Select Sensors to Visualize", 
                                      options=sensor_cols,
                                      default=sensor_cols[:3] if len(sensor_cols) > 3 else sensor_cols)
    
    if selected_sensors:
        sensor_fig = plot_sensor_data(data, unit_id=selected_unit, sensors=selected_sensors)
        st.plotly_chart(sensor_fig, use_container_width=True)
    else:
        st.info("Please select at least one sensor to visualize.")
else:
    st.warning("No sensor columns found in the data.")

# RUL Visualization
if 'RUL' in data.columns:
    st.subheader("Remaining Useful Life (RUL) over Time")
    rul_fig = plot_rul_vs_cycles(data, unit_id=selected_unit)
    st.plotly_chart(rul_fig, use_container_width=True)

# Correlation matrix
st.subheader("Feature Correlation")
# Select columns for correlation (exclude non-numeric)
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
# Limit to sensors and RUL for clarity
if 'sensor_1' in data.columns:
    corr_cols = [col for col in numeric_cols if 'sensor' in col or col == 'RUL']
    corr_cols = corr_cols[:15] if len(corr_cols) > 15 else corr_cols  # Limit for better visualization
else:
    corr_cols = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols

# Create correlation matrix
if corr_cols:
    corr_matrix = data[corr_cols].corr()
    
    # Plot with plotly
    corr_fig = px.imshow(corr_matrix, 
                         text_auto='.2f',
                         color_continuous_scale='RdBu_r',
                         title="Correlation Matrix")
    st.plotly_chart(corr_fig, use_container_width=True)
