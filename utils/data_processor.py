import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import io
import boto3

def load_sample_data():
    """
    Load sample NASA turbofan engine degradation data or generate synthetic data
    if the real data is not available.
    """
    try:
        # NASA turbofan dataset structure (simplified)
        # Create a sample dataset from FD001.txt from the Turbofan dataset
        # This simulates loading from a file or S3 bucket
        columns = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                  [f'sensor_{i}' for i in range(1, 22)]
        
        # Generate synthetic data (this would normally come from an S3 bucket)
        np.random.seed(42)  # For reproducibility
        n_samples = 1000
        n_units = 20
        
        data = []
        for unit in range(1, n_units + 1):
            max_cycles = np.random.randint(128, 356)  # Random failure point
            for cycle in range(1, max_cycles + 1):
                # Operating settings
                op_settings = np.random.normal(0, 1, 3)
                
                # Sensor readings - some with trends as unit degrades
                sensors = []
                for i in range(21):
                    # Add some degradation trend for certain sensors
                    if i in [2, 3, 4, 7, 11, 15]:
                        base = np.random.normal(0, 1)
                        trend = 0.1 * (cycle / max_cycles) * np.random.uniform(0.5, 1.5)
                        sensors.append(base + trend)
                    else:
                        sensors.append(np.random.normal(0, 1))
                
                # Calculate RUL
                rul = max_cycles - cycle
                
                row = [unit, cycle] + list(op_settings) + sensors + [rul]
                data.append(row)
        
        # Create dataframe
        df = pd.DataFrame(data, columns=columns + ['RUL'])
        return df
        
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        return None

def load_data_from_s3(bucket_name, file_key):
    """
    Load data from an S3 bucket
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read()
        
        # Determine file type and read accordingly
        if file_key.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file_key.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(content))
        else:
            # Assuming space-delimited text file (like NASA dataset)
            df = pd.read_csv(io.BytesIO(content), delim_whitespace=True, header=None)
            
        return df
    except Exception as e:
        print(f"Error loading data from S3: {str(e)}")
        return None

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data:
    - Handle missing values
    - Engineer features
    - Scale numerical features
    - Split into train/test sets
    """
    if df is None:
        return None, None, None, None, None
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    data = data.fillna(method='ffill')  # Forward fill
    data = data.fillna(method='bfill')  # Backward fill for any remaining NAs
    
    # Feature engineering for time series data
    # 1. Create rolling statistics for sensors
    sensor_cols = [col for col in data.columns if 'sensor' in col]
    for col in sensor_cols:
        # Rolling mean with window 5
        data[f'{col}_rolling_mean5'] = data.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean())
        
        # Rolling std with window 5
        data[f'{col}_rolling_std5'] = data.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))
    
    # 2. Create difference features
    for col in sensor_cols:
        data[f'{col}_diff'] = data.groupby('unit_number')[col].diff().fillna(0)
    
    # Prepare features and target
    X = data.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y = data['RUL']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler

def engineer_features(df):
    """
    Engineer new features from the raw data
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    
    # Extract sensor columns
    sensor_cols = [col for col in data.columns if 'sensor' in col]
    
    # Feature engineering
    # 1. Rolling statistics
    for col in sensor_cols:
        # Rolling mean with window sizes 5, 10, 15
        for window in [5, 10, 15]:
            data[f'{col}_rolling_mean{window}'] = data.groupby('unit_number')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Rolling std with window size 5
        data[f'{col}_rolling_std5'] = data.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))
    
    # 2. Differences and rates of change
    for col in sensor_cols:
        # First difference
        data[f'{col}_diff1'] = data.groupby('unit_number')[col].diff().fillna(0)
        
        # Second difference
        data[f'{col}_diff2'] = data.groupby('unit_number')[f'{col}_diff1'].diff().fillna(0)
        
    # 3. Sensor interactions
    # Pairwise ratios between selected sensors
    important_sensors = sensor_cols[:5]  # First 5 sensors for example
    for i, sensor1 in enumerate(important_sensors):
        for sensor2 in important_sensors[i+1:]:
            # Avoid division by zero
            data[f'{sensor1}_div_{sensor2}'] = data[sensor1] / (data[sensor2] + 1e-8)
    
    return data
