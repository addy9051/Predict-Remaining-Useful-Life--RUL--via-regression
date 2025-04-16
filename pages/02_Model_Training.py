import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle
import os
import time

from utils.data_processor import load_sample_data, preprocess_data
from utils.model_training import (
    train_random_forest, 
    train_gradient_boosting, 
    hyperparameter_tuning,
    save_model_locally
)

# Set page title
st.set_page_config(page_title="Model Training", page_icon="🔧")

# Header
st.title("Model Training")
st.markdown("Train regression models to predict Remaining Useful Life (RUL).")

# Initialize session state for models
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'gb_model' not in st.session_state:
    st.session_state.gb_model = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'train_metrics' not in st.session_state:
    st.session_state.train_metrics = {}

# Get data
from utils.data_processor import load_sample_data, load_data_from_s3

# Check if data exists in session state, otherwise load sample data
if 'data' not in st.session_state or st.session_state.data is None:
    st.session_state.data = load_sample_data()
    
data = st.session_state.data

if data is None:
    st.error("No data available. Please load data from the Data Exploration page.")
    st.stop()

# Show data preview
st.subheader("Data Preview")
st.dataframe(data.head())

# Data preprocessing
st.header("Data Preprocessing")

# Split ratio for train/test
test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# Random state for reproducibility
random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, step=1)

# Preprocess button
if st.button("Preprocess Data"):
    with st.spinner("Preprocessing data..."):
        # Check if 'RUL' column exists
        if 'RUL' not in data.columns:
            st.error("The data does not contain a 'RUL' column for prediction.")
            st.stop()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            data, test_size=test_size, random_state=random_state
        )
        
        # Store in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        
        # Get feature names (excluding 'RUL', 'unit_number', 'time_cycles')
        feature_cols = [col for col in data.columns if col not in ['RUL', 'unit_number', 'time_cycles']]
        st.session_state.feature_names = feature_cols
        
        st.success(f"Data preprocessed! Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Model Training Section
st.header("Model Training")

# Check if data is preprocessed
if st.session_state.X_train is None:
    st.info("Please preprocess the data first.")
else:
    # Model Training Tabs
    model_tab1, model_tab2, model_tab3 = st.tabs(["Random Forest", "Gradient Boosting", "Hyperparameter Tuning"])
    
    with model_tab1:
        st.subheader("Random Forest Regressor")
        
        # Hyperparameters
        rf_n_estimators = st.slider("Number of Estimators", min_value=10, max_value=300, value=100, step=10)
        rf_max_depth = st.slider("Max Depth", min_value=5, max_value=50, value=20, step=5)
        rf_min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)
        
        # Train button
        if st.button("Train Random Forest"):
            with st.spinner("Training Random Forest model..."):
                # Configure parameters
                rf_params = {
                    'n_estimators': rf_n_estimators,
                    'max_depth': rf_max_depth,
                    'min_samples_split': rf_min_samples_split,
                    'random_state': random_state
                }
                
                # Train model
                start_time = time.time()
                rf_model = train_random_forest(
                    st.session_state.X_train, 
                    st.session_state.y_train, 
                    params=rf_params
                )
                training_time = time.time() - start_time
                
                # Make predictions and evaluate
                y_pred_train = rf_model.predict(st.session_state.X_train)
                y_pred_test = rf_model.predict(st.session_state.X_test)
                
                train_rmse = np.sqrt(np.mean((y_pred_train - st.session_state.y_train) ** 2))
                test_rmse = np.sqrt(np.mean((y_pred_test - st.session_state.y_test) ** 2))
                
                # Store model and metrics
                st.session_state.rf_model = rf_model
                st.session_state.train_metrics['random_forest'] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'training_time': training_time,
                    'params': rf_params,
                    'model': rf_model
                }
                
                # Save model locally
                save_model_locally(rf_model, 'models/random_forest.joblib')
                
                # Update best model if applicable
                if st.session_state.best_model is None or test_rmse < st.session_state.train_metrics.get(
                    st.session_state.best_model_name, {}).get('test_rmse', float('inf')):
                    st.session_state.best_model = rf_model
                    st.session_state.best_model_name = 'random_forest'
                
                # Display results
                st.success(f"Random Forest model trained in {training_time:.2f} seconds!")
                st.write(f"Training RMSE: {train_rmse:.4f}")
                st.write(f"Test RMSE: {test_rmse:.4f}")
                
                # Feature importance (Top 10)
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.write("Top 10 Feature Importance:")
                st.bar_chart(feature_importance.set_index('Feature'))
    
    with model_tab2:
        st.subheader("Gradient Boosting Regressor")
        
        # Hyperparameters
        gb_n_estimators = st.slider("Number of Estimators (GB)", min_value=10, max_value=300, value=100, step=10)
        gb_learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
        gb_max_depth = st.slider("Max Depth (GB)", min_value=2, max_value=10, value=5, step=1)
        
        # Train button
        if st.button("Train Gradient Boosting"):
            with st.spinner("Training Gradient Boosting model..."):
                # Configure parameters
                gb_params = {
                    'n_estimators': gb_n_estimators,
                    'learning_rate': gb_learning_rate,
                    'max_depth': gb_max_depth,
                    'random_state': random_state
                }
                
                # Train model
                start_time = time.time()
                gb_model = train_gradient_boosting(
                    st.session_state.X_train, 
                    st.session_state.y_train, 
                    params=gb_params
                )
                training_time = time.time() - start_time
                
                # Make predictions and evaluate
                y_pred_train = gb_model.predict(st.session_state.X_train)
                y_pred_test = gb_model.predict(st.session_state.X_test)
                
                train_rmse = np.sqrt(np.mean((y_pred_train - st.session_state.y_train) ** 2))
                test_rmse = np.sqrt(np.mean((y_pred_test - st.session_state.y_test) ** 2))
                
                # Store model and metrics
                st.session_state.gb_model = gb_model
                st.session_state.train_metrics['gradient_boosting'] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'training_time': training_time,
                    'params': gb_params,
                    'model': gb_model
                }
                
                # Save model locally
                save_model_locally(gb_model, 'models/gradient_boosting.joblib')
                
                # Update best model if applicable
                if st.session_state.best_model is None or test_rmse < st.session_state.train_metrics.get(
                    st.session_state.best_model_name, {}).get('test_rmse', float('inf')):
                    st.session_state.best_model = gb_model
                    st.session_state.best_model_name = 'gradient_boosting'
                
                # Display results
                st.success(f"Gradient Boosting model trained in {training_time:.2f} seconds!")
                st.write(f"Training RMSE: {train_rmse:.4f}")
                st.write(f"Test RMSE: {test_rmse:.4f}")
                
                # Feature importance (Top 10)
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Importance': gb_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.write("Top 10 Feature Importance:")
                st.bar_chart(feature_importance.set_index('Feature'))
    
    with model_tab3:
        st.subheader("Hyperparameter Tuning")
        
        # Model selection
        model_type = st.radio("Select Model for Tuning", ["Random Forest", "Gradient Boosting"])
        
        # Tune button
        if st.button("Tune Hyperparameters"):
            with st.spinner(f"Tuning {model_type} model (this may take several minutes)..."):
                # Convert to internal model type format
                model_type_internal = 'rf' if model_type == "Random Forest" else 'gb'
                
                # Run hyperparameter tuning
                start_time = time.time()
                best_params, best_model = hyperparameter_tuning(
                    st.session_state.X_train, 
                    st.session_state.y_train, 
                    model_type=model_type_internal,
                    cv=3  # Limited cross-validation for speed
                )
                tuning_time = time.time() - start_time
                
                # Evaluate model
                y_pred_train = best_model.predict(st.session_state.X_train)
                y_pred_test = best_model.predict(st.session_state.X_test)
                
                train_rmse = np.sqrt(np.mean((y_pred_train - st.session_state.y_train) ** 2))
                test_rmse = np.sqrt(np.mean((y_pred_test - st.session_state.y_test) ** 2))
                
                # Store tuned model and metrics
                model_name = 'tuned_rf' if model_type == "Random Forest" else 'tuned_gb'
                st.session_state.train_metrics[model_name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'training_time': tuning_time,
                    'params': best_params,
                    'model': best_model
                }
                
                # Save model locally
                save_model_locally(best_model, f'models/{model_name}.joblib')
                
                # Update best model if applicable
                if st.session_state.best_model is None or test_rmse < st.session_state.train_metrics.get(
                    st.session_state.best_model_name, {}).get('test_rmse', float('inf')):
                    st.session_state.best_model = best_model
                    st.session_state.best_model_name = model_name
                
                # Display results
                st.success(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds!")
                st.write("Best Parameters:")
                st.json(best_params)
                st.write(f"Training RMSE: {train_rmse:.4f}")
                st.write(f"Test RMSE: {test_rmse:.4f}")
                
                # Feature importance (Top 10)
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.write("Top 10 Feature Importance:")
                    st.bar_chart(feature_importance.set_index('Feature'))

    # Model Comparison
    st.header("Model Comparison")
    
    if st.session_state.train_metrics:
        comparison_data = []
        for model_name, metrics in st.session_state.train_metrics.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Training RMSE': metrics['train_rmse'],
                'Test RMSE': metrics['test_rmse'],
                'Training Time (s)': metrics['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df)
        
        # Highlight best model
        best_model_name = st.session_state.best_model_name
        if best_model_name:
            formatted_name = best_model_name.replace('_', ' ').title()
            st.success(f"Best Model: {formatted_name} (Test RMSE: {st.session_state.train_metrics[best_model_name]['test_rmse']:.4f})")
    else:
        st.info("Train at least one model to see the comparison.")
