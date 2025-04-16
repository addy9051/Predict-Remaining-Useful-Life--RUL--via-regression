import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import boto3
import io
import pickle

def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Dictionary of hyperparameters (optional)
        
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, params=None):
    """
    Train a Gradient Boosting regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Dictionary of hyperparameters (optional)
        
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train, model_type='rf', cv=5):
    """
    Tune hyperparameters with GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'rf' for Random Forest, 'gb' for Gradient Boosting
        cv: Number of cross-validation folds
        
    Returns:
        Best parameters and best estimator
    """
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gb':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        raise ValueError("model_type must be 'rf' or 'gb'")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actual': y_test
    }

def save_model_to_s3(model, bucket_name, model_key):
    """
    Save model to S3 bucket
    
    Args:
        model: Trained model to save
        bucket_name: Name of the S3 bucket
        model_key: Key to save the model under
        
    Returns:
        Boolean indicating success
    """
    try:
        # Serialize model to bytes
        model_bytes = io.BytesIO()
        pickle.dump(model, model_bytes)
        model_bytes.seek(0)
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(model_bytes, bucket_name, model_key)
        return True
    except Exception as e:
        print(f"Error saving model to S3: {str(e)}")
        return False

def load_model_from_s3(bucket_name, model_key):
    """
    Load model from S3 bucket
    
    Args:
        bucket_name: Name of the S3 bucket
        model_key: Key where the model is stored
        
    Returns:
        Loaded model or None if error
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model_bytes = response['Body'].read()
        
        # Deserialize model
        model = pickle.loads(model_bytes)
        return model
    except Exception as e:
        print(f"Error loading model from S3: {str(e)}")
        return None

def save_model_locally(model, file_path):
    """
    Save model to local filesystem
    
    Args:
        model: Trained model to save
        file_path: Path to save the model
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model
        joblib.dump(model, file_path)
        return True
    except Exception as e:
        print(f"Error saving model locally: {str(e)}")
        return False

def load_model_locally(file_path):
    """
    Load model from local filesystem
    
    Args:
        file_path: Path where the model is stored
        
    Returns:
        Loaded model or None if error
    """
    try:
        if os.path.exists(file_path):
            model = joblib.load(file_path)
            return model
        else:
            print(f"Model file not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading model locally: {str(e)}")
        return None
