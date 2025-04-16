import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_processor import load_sample_data, load_data_from_s3
from utils.aws_utils import list_s3_buckets, list_s3_objects
from utils.visualization import plot_sensor_data, plot_rul_vs_cycles

# Set page title
st.set_page_config(page_title="Data Exploration", page_icon="📊")

# Header
st.title("Data Exploration")
st.markdown("Explore and analyze the machine sensor data for predictive maintenance.")

# Sidebar for data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Sample Data", "AWS S3 Bucket"]
)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

# Load data based on selection
if data_source == "Sample Data":
    # Load sample data
    if st.session_state.data is None:
        st.session_state.data = load_sample_data()
    
    data = st.session_state.data
    
    if data is not None:
        st.success("Sample data loaded successfully!")
    else:
        st.error("Failed to load sample data.")
        st.stop()

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
            if st.sidebar.button("Load Data"):
                with st.spinner("Loading data from S3..."):
                    data = load_data_from_s3(selected_bucket, selected_object)
                    
                    if data is not None:
                        st.session_state.data = data
                        st.success(f"Data loaded from S3: {selected_object}")
                    else:
                        st.error("Failed to load data from S3.")
                        st.stop()
            else:
                # Use existing data or sample data
                data = st.session_state.data if st.session_state.data is not None else load_sample_data()
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
