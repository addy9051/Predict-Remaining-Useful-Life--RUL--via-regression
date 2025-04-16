import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import datetime
import json
import boto3
from io import StringIO

from utils.data_processor import load_sample_data
from utils.aws_utils import upload_to_s3

# Set page title
st.set_page_config(page_title="Make Predictions", page_icon="🔮")

# Header
st.title("RUL Predictions")
st.markdown("Make predictions with trained models and input data.")

# Check if models are available
if ('rf_model' not in st.session_state or st.session_state.rf_model is None) and \
   ('gb_model' not in st.session_state or st.session_state.gb_model is None) and \
   ('best_model' not in st.session_state or st.session_state.best_model is None):
    st.warning("No trained models available. Please train models in the Model Training page first.")
    st.stop()

# Check if scaler is available
if 'scaler' not in st.session_state or st.session_state.scaler is None:
    st.warning("No data scaler available. Please preprocess data in the Model Training page first.")
    st.stop()

# Input Methods
st.header("Input Methods")
input_method = st.radio(
    "Select Input Method",
    ["Sample Data", "CSV Upload", "Manual Input"]
)

# Function to preprocess new data
def preprocess_new_data(new_data):
    """Preprocess new data using the stored scaler"""
    # Get model columns
    model_columns = st.session_state.feature_names
    
    # Engineer the same features that were used during training if they're missing
    # For example: create rolling means, diffs, etc. if they're in model_columns
    engineered_data = new_data.copy()
    
    # Generate derived features if needed
    for col in model_columns:
        # Handle rolling mean features
        if '_rolling_mean' in col:
            base_col = col.split('_rolling_mean')[0]
            window = int(col.split('_rolling_mean')[1]) if col.split('_rolling_mean')[1] else 5
            
            if base_col in engineered_data.columns and col not in engineered_data.columns:
                engineered_data[col] = engineered_data[base_col].rolling(window=window, min_periods=1).mean()
        
        # Handle rolling std features
        elif '_rolling_std' in col:
            base_col = col.split('_rolling_std')[0]
            window = int(col.split('_rolling_std')[1]) if col.split('_rolling_std')[1] else 5
            
            if base_col in engineered_data.columns and col not in engineered_data.columns:
                engineered_data[col] = engineered_data[base_col].rolling(window=window, min_periods=1).std()
        
        # Handle diff features
        elif '_diff' in col:
            base_col = col.split('_diff')[0]
            
            if base_col in engineered_data.columns and col not in engineered_data.columns:
                engineered_data[col] = engineered_data[base_col].diff().fillna(0)
    
    # Check if any required columns are still missing
    missing_cols = [col for col in model_columns if col not in engineered_data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols[:10])}..." if len(missing_cols) > 10 else f"Missing required columns: {', '.join(missing_cols)}")
        return None
    
    # Select only the columns used for training
    data_for_prediction = engineered_data[model_columns].copy()
    
    # Apply the same preprocessing as during training
    data_for_prediction = data_for_prediction.fillna(method='ffill')
    data_for_prediction = data_for_prediction.fillna(method='bfill')
    
    # Scale the data
    scaled_data = st.session_state.scaler.transform(data_for_prediction)
    
    return scaled_data

# Function to make predictions
def make_prediction(input_data, model_name):
    """Make prediction with selected model"""
    # Get the model
    if model_name == "Random Forest":
        model = st.session_state.rf_model
    elif model_name == "Gradient Boosting":
        model = st.session_state.gb_model
    else:  # Best model
        model = st.session_state.best_model
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions

# Model selection
models = []
if 'rf_model' in st.session_state and st.session_state.rf_model is not None:
    models.append('Random Forest')
if 'gb_model' in st.session_state and st.session_state.gb_model is not None:
    models.append('Gradient Boosting')
if 'best_model' in st.session_state and st.session_state.best_model is not None and 'best_model_name' in st.session_state:
    models.append(f'Best Model ({st.session_state.best_model_name.replace("_", " ").title()})')

selected_model_name = st.selectbox("Select Model for Prediction", models)

# Handle different input methods
if input_method == "Sample Data":
    # Load sample data
    sample_data = load_sample_data()
    
    if sample_data is None:
        st.error("Could not load sample data.")
        st.stop()
    
    # Allow selecting a subset
    num_samples = st.slider("Number of samples", min_value=1, max_value=min(100, len(sample_data)), value=10)
    
    # Get random samples
    if 'unit_number' in sample_data.columns:
        # Get unique units
        unique_units = sample_data['unit_number'].unique()
        selected_unit = st.selectbox("Select unit", unique_units)
        
        # Filter by unit
        unit_data = sample_data[sample_data['unit_number'] == selected_unit]
        
        # Get the last n cycles for this unit
        if len(unit_data) > num_samples:
            prediction_data = unit_data.tail(num_samples).reset_index(drop=True)
        else:
            prediction_data = unit_data.reset_index(drop=True)
    else:
        # Just get random samples
        prediction_data = sample_data.sample(num_samples).reset_index(drop=True)
    
    # Show selected data
    st.subheader("Selected Data for Prediction")
    st.dataframe(prediction_data)
    
    # Process the data for prediction
    processed_data = preprocess_new_data(prediction_data)
    
    if processed_data is not None:
        # Make prediction when button is clicked
        if st.button("Make Prediction"):
            with st.spinner("Making predictions..."):
                # Make predictions
                predictions = make_prediction(processed_data, selected_model_name)
                
                # Show predictions
                st.subheader("Predictions")
                
                # Create result dataframe
                result_df = pd.DataFrame({
                    'Predicted RUL': predictions.round(2)
                })
                
                # Add actual RUL if available
                if 'RUL' in prediction_data.columns:
                    result_df['Actual RUL'] = prediction_data['RUL'].values
                    result_df['Error'] = (result_df['Predicted RUL'] - result_df['Actual RUL']).round(2)
                
                # Add unit and cycle info if available
                if 'unit_number' in prediction_data.columns:
                    result_df['Unit'] = prediction_data['unit_number'].values
                
                if 'time_cycles' in prediction_data.columns:
                    result_df['Cycle'] = prediction_data['time_cycles'].values
                
                # Display results
                st.dataframe(result_df)
                
                # Plot if actual values are available
                if 'Actual RUL' in result_df.columns:
                    comparison_fig = px.bar(result_df, x=result_df.index, 
                                           y=['Actual RUL', 'Predicted RUL'],
                                           barmode='group',
                                           title="Predicted vs Actual RUL")
                    st.plotly_chart(comparison_fig)
                else:
                    # Just plot predictions
                    pred_fig = px.bar(result_df, x=result_df.index, 
                                     y='Predicted RUL',
                                     title="Predicted RUL")
                    st.plotly_chart(pred_fig)
                
                # Save predictions to CSV
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                result_df.to_csv(f"predictions_{timestamp}.csv", index=False)
                
                st.success(f"Predictions saved to predictions_{timestamp}.csv")
                
                # Option to upload to S3
                if st.button("Upload Predictions to S3"):
                    # Get S3 bucket name
                    bucket_name = st.text_input("S3 Bucket Name")
                    
                    if bucket_name:
                        try:
                            # Convert to CSV
                            csv_buffer = StringIO()
                            result_df.to_csv(csv_buffer, index=False)
                            
                            # Upload to S3
                            s3_client = boto3.client('s3')
                            s3_client.put_object(
                                Bucket=bucket_name,
                                Key=f"predictions/predictions_{timestamp}.csv",
                                Body=csv_buffer.getvalue()
                            )
                            
                            st.success(f"Predictions uploaded to S3: {bucket_name}/predictions/predictions_{timestamp}.csv")
                        except Exception as e:
                            st.error(f"Error uploading to S3: {str(e)}")
                    else:
                        st.warning("Please enter an S3 bucket name.")

elif input_method == "CSV Upload":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            input_data = pd.read_csv(uploaded_file)
            
            # Show the uploaded data
            st.subheader("Uploaded Data")
            st.dataframe(input_data.head())
            
            # Allow column mapping if needed
            st.subheader("Column Mapping")
            st.info("If your column names differ from the required ones, map them here.")
            
            # Show required columns
            required_cols = st.session_state.feature_names
            st.write("Required columns:")
            st.write(", ".join(required_cols[:10]) + ("..." if len(required_cols) > 10 else ""))
            
            # Ask if mapping is needed
            needs_mapping = st.checkbox("I need to map columns")
            
            if needs_mapping:
                # Create a mapping dictionary
                mapping = {}
                
                # For simplicity, let user map the most important features
                important_features = required_cols[:5]  # Map first 5 for simplicity
                
                for col in important_features:
                    mapped_col = st.selectbox(f"Map '{col}' to:", [""] + list(input_data.columns), key=f"map_{col}")
                    if mapped_col:
                        mapping[col] = mapped_col
                
                # Apply mapping
                if mapping:
                    # Create a copy with renamed columns
                    mapped_data = input_data.copy()
                    for target, source in mapping.items():
                        mapped_data[target] = input_data[source]
                    
                    # Now try to process this data
                    processed_data = preprocess_new_data(mapped_data)
                else:
                    st.warning("Please map at least one column or uncheck the mapping option.")
                    processed_data = None
            else:
                # Try to process without mapping
                processed_data = preprocess_new_data(input_data)
            
            if processed_data is not None:
                # Make prediction when button is clicked
                if st.button("Make Prediction"):
                    with st.spinner("Making predictions..."):
                        # Make predictions
                        predictions = make_prediction(processed_data, selected_model_name)
                        
                        # Show predictions
                        st.subheader("Predictions")
                        
                        # Create result dataframe
                        result_df = pd.DataFrame({
                            'Predicted RUL': predictions.round(2)
                        })
                        
                        # Add unit and cycle info if available
                        if 'unit_number' in input_data.columns:
                            result_df['Unit'] = input_data['unit_number'].values
                        
                        if 'time_cycles' in input_data.columns:
                            result_df['Cycle'] = input_data['time_cycles'].values
                        
                        # Display results
                        st.dataframe(result_df)
                        
                        # Plot predictions
                        pred_fig = px.bar(result_df, x=result_df.index, 
                                         y='Predicted RUL',
                                         title="Predicted RUL")
                        st.plotly_chart(pred_fig)
                        
                        # Save predictions to CSV
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_df.to_csv(f"predictions_{timestamp}.csv", index=False)
                        
                        st.success(f"Predictions saved to predictions_{timestamp}.csv")
        
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")

elif input_method == "Manual Input":
    st.subheader("Manual Input")
    st.info("Enter values for key features to predict RUL")
    
    # Get sample data structure
    if 'data' in st.session_state and st.session_state.data is not None:
        sample = st.session_state.data
    else:
        sample = load_sample_data()
        
    if sample is None:
        st.error("Could not load sample data structure. Please try another input method.")
        st.stop()
    
    # Get sensor columns (limit to top 10 for usability)
    sensor_cols = [col for col in sample.columns if 'sensor' in col][:10]
    
    # Create input form
    manual_data = {}
    
    # Unit ID and cycle (optional but useful for context)
    unit_id = st.number_input("Unit ID", min_value=1, value=1)
    manual_data['unit_number'] = unit_id
    
    cycle = st.number_input("Cycle", min_value=1, value=100)
    manual_data['time_cycles'] = cycle
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Add sensor inputs
    for i, col in enumerate(sensor_cols):
        # Alternate between columns
        input_col = col1 if i % 2 == 0 else col2
        
        # Get sample range for this sensor
        if sample is not None:
            min_val = float(sample[col].min())
            max_val = float(sample[col].max())
            mean_val = float(sample[col].mean())
            
            # Create slider with appropriate range
            val = input_col.slider(
                f"{col}", 
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                key=f"manual_{col}"
            )
        else:
            # Fallback if no sample data
            val = input_col.number_input(f"{col}", value=0.0, key=f"manual_{col}")
        
        manual_data[col] = val
    
    # Create DataFrame from manual inputs
    manual_df = pd.DataFrame([manual_data])
    
    # Show the input data
    st.subheader("Input Data")
    st.dataframe(manual_df)
    
    # Process the data for prediction
    # We need to handle missing columns that might be required by the model
    for col in st.session_state.feature_names:
        if col not in manual_df.columns:
            # Fill with zeros for missing columns
            manual_df[col] = 0.0
    
    processed_data = preprocess_new_data(manual_df)
    
    if processed_data is not None:
        # Make prediction when button is clicked
        if st.button("Make Prediction"):
            with st.spinner("Making predictions..."):
                # Make predictions
                predictions = make_prediction(processed_data, selected_model_name)
                
                # Show predictions
                st.subheader("Prediction Result")
                
                # Create result card
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    st.metric("Predicted RUL", f"{predictions[0]:.2f} cycles")
                
                with result_col2:
                    # Add interpretation
                    if predictions[0] < 50:
                        st.error("⚠️ Maintenance needed soon!")
                    elif predictions[0] < 100:
                        st.warning("⚠️ Plan maintenance in medium term")
                    else:
                        st.success("✅ Healthy, long remaining life")
                
                # Save this prediction to history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'unit_id': unit_id,
                    'cycle': cycle,
                    'predicted_rul': float(predictions[0]),
                    'model': selected_model_name
                })
                
                # Show prediction history
                if st.session_state.prediction_history:
                    st.subheader("Prediction History")
                    history_df = pd.DataFrame(st.session_state.prediction_history)
                    st.dataframe(history_df)
                    
                    # Plot history
                    if len(history_df) > 1:
                        history_fig = px.line(history_df, x='timestamp', y='predicted_rul',
                                            title="Prediction History")
                        st.plotly_chart(history_fig)
