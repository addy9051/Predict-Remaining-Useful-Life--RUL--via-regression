import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import io
import boto3
import requests
import json
import time
from datetime import datetime
import glob

def load_nasa_cmapss_data(dataset='FD001', data_dir='./data'):
    """
    Load the NASA CMAPSS Turbofan Engine Degradation dataset
    
    Args:
        dataset: Dataset ID ('FD001', 'FD002', 'FD003', or 'FD004')
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame containing the processed dataset with RUL values
    """
    try:
        print(f"Loading NASA CMAPSS {dataset} dataset...")
        
        # Paths to data files
        train_file = os.path.join(data_dir, f'train_{dataset}.txt')
        test_file = os.path.join(data_dir, f'test_{dataset}.txt')
        rul_file = os.path.join(data_dir, f'RUL_{dataset}.txt')
        
        # Check if files exist
        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(rul_file)):
            print(f"Data files for {dataset} not found in {data_dir}")
            return None
        
        # Column names according to the dataset description
        columns = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                  [f'sensor_{i}' for i in range(1, 22)]
        
        # Load training data
        train_df = pd.read_csv(train_file, delimiter=' ', header=None, names=columns)
        train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
        
        # Load test data
        test_df = pd.read_csv(test_file, delimiter=' ', header=None, names=columns)
        test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
        
        # Load RUL values for test data
        rul_values = pd.read_csv(rul_file, header=None).values.flatten()
        
        # Calculate RUL for training data
        # For each unit, max cycle = failure point, so RUL = max_cycle - current_cycle
        train_max_cycles = train_df.groupby('unit_number')['time_cycles'].max().reset_index()
        train_max_cycles.columns = ['unit_number', 'max_cycles']
        train_df = pd.merge(train_df, train_max_cycles, on='unit_number')
        train_df['RUL'] = train_df['max_cycles'] - train_df['time_cycles']
        train_df = train_df.drop('max_cycles', axis=1)
        
        # Calculate RUL for test data
        # First, find the max cycle for each unit in test data
        test_max_cycles = test_df.groupby('unit_number')['time_cycles'].max().reset_index()
        test_max_cycles.columns = ['unit_number', 'max_cycles']
        
        # Add a RUL column based on the true RUL values
        # The RUL value in the file corresponds to the last cycle of each unit
        test_rul_df = pd.DataFrame({
            'unit_number': range(1, len(rul_values) + 1),
            'true_rul': rul_values
        })
        
        # Merge max cycles and RUL values
        test_max_rul = pd.merge(test_max_cycles, test_rul_df, on='unit_number')
        
        # Merge with test data
        test_df = pd.merge(test_df, test_max_rul, on='unit_number')
        
        # Calculate RUL for each record in test data
        # RUL at the last cycle = true_rul, so RUL at earlier cycles is higher by the difference
        test_df['RUL'] = test_df['true_rul'] + (test_df['max_cycles'] - test_df['time_cycles'])
        
        # Drop helper columns
        test_df = test_df.drop(['max_cycles', 'true_rul'], axis=1)
        
        # Combine train and test data
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        print(f"Successfully loaded NASA CMAPSS {dataset} dataset. Total records: {len(combined_df)}")
        return combined_df
    
    except Exception as e:
        print(f"Error loading NASA CMAPSS data: {str(e)}")
        return None

def load_sample_data():
    """
    Load NASA CMAPSS turbofan engine degradation data if available,
    otherwise generate synthetic data.
    """
    # First, try to load the real NASA data
    nasa_data = load_nasa_cmapss_data(dataset='FD001')
    if nasa_data is not None:
        return nasa_data
    
    # If NASA data is not available, generate synthetic data as fallback
    try:
        print("Generating synthetic data as fallback...")
        # NASA turbofan dataset structure (simplified)
        columns = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                  [f'sensor_{i}' for i in range(1, 22)]
        
        # Generate synthetic data
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
        print(f"Error generating sample data: {str(e)}")
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

def load_data_from_api(api_url, api_key=None, dataset_type='turbofan', retries=3, timeout=10):
    """
    Load data from a REST API endpoint
    
    Args:
        api_url: URL of the API endpoint
        api_key: Optional API key for authentication
        dataset_type: Type of dataset to fetch ('turbofan', 'sensor', or 'custom')
        retries: Number of retry attempts in case of connection issues
        timeout: Request timeout in seconds
    
    Returns:
        pandas DataFrame with the data or None if error
    """
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Add retry logic for robust API connections
    for attempt in range(retries):
        try:
            print(f"Attempting to fetch data from API (attempt {attempt+1}/{retries})...")
            
            # Make the API request
            response = requests.get(api_url, headers=headers, timeout=timeout)
            
            # Check if request was successful
            if response.status_code == 200:
                # Parse the data based on dataset type
                if dataset_type == 'turbofan':
                    return _parse_turbofan_api_data(response.json())
                elif dataset_type == 'sensor':
                    return _parse_sensor_api_data(response.json())
                else:
                    # Try to parse as generic JSON
                    return _parse_generic_api_data(response.json())
            else:
                print(f"API request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                
                # Wait before retry (exponential backoff)
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API: {str(e)}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    print("Failed to fetch data from API after multiple attempts.")
    return None

def _parse_turbofan_api_data(json_data):
    """Parse data specifically formatted for turbofan engine data"""
    try:
        # Extract data from the JSON response
        # This assumes a specific structure that matches our needs
        if isinstance(json_data, dict) and 'data' in json_data:
            # If data is in a nested 'data' field
            data_list = json_data['data']
        elif isinstance(json_data, list):
            # If data is directly a list
            data_list = json_data
        else:
            print("Unexpected JSON structure")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Check if we have necessary columns
        required_cols = ['unit_number', 'time_cycles']
        sensor_pattern = 'sensor_'
        
        # If columns are named differently, try to map them
        if not all(col in df.columns for col in required_cols):
            # Try common alternative names
            column_mapping = {
                'unit': 'unit_number',
                'unit_id': 'unit_number',
                'engine': 'unit_number',
                'engine_id': 'unit_number',
                'cycle': 'time_cycles',
                'cycles': 'time_cycles',
                'time': 'time_cycles'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Check for sensor columns and rename if needed
        sensor_cols = [col for col in df.columns if sensor_pattern in col.lower()]
        if not sensor_cols:
            # Try to identify sensor columns by other patterns
            potential_sensor_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['measurement', 'reading', 'sensor'])]
            
            # If we found potential sensor columns, rename them
            if potential_sensor_cols:
                for i, col in enumerate(potential_sensor_cols):
                    df[f'sensor_{i+1}'] = df[col]
        
        # Calculate RUL if not present
        if 'RUL' not in df.columns:
            # Group by unit_number and calculate max cycle for each unit
            if 'unit_number' in df.columns and 'time_cycles' in df.columns:
                max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
                max_cycles.columns = ['unit_number', 'max_cycles']
                
                # Merge with the main DataFrame
                df = pd.merge(df, max_cycles, on='unit_number')
                
                # Calculate RUL
                df['RUL'] = df['max_cycles'] - df['time_cycles']
                
                # Drop the temporary column
                df = df.drop('max_cycles', axis=1)
        
        return df
    
    except Exception as e:
        print(f"Error parsing turbofan API data: {str(e)}")
        return None

def _parse_sensor_api_data(json_data):
    """Parse data from a general sensor data API"""
    try:
        # Handle different possible JSON structures
        if isinstance(json_data, dict):
            if 'data' in json_data:
                # Case: {"data": [...]}
                data = json_data['data']
            elif 'readings' in json_data:
                # Case: {"readings": [...]}
                data = json_data['readings']
            elif 'results' in json_data:
                # Case: {"results": [...]}
                data = json_data['results']
            else:
                # Case: JSON is directly the data structure we need
                data = [json_data]
        elif isinstance(json_data, list):
            # Case: JSON is a list of records
            data = json_data
        else:
            print(f"Unexpected API response format: {type(json_data)}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check for timestamp column and convert to datetime if needed
        time_columns = [col for col in df.columns if any(time_indicator in col.lower() 
                                                        for time_indicator in ['time', 'date', 'timestamp'])]
        
        if time_columns:
            # Use the first identified time column
            time_col = time_columns[0]
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except:
                # If conversion fails, leave as is
                pass
            
            # Sort by timestamp
            df = df.sort_values(by=time_col).reset_index(drop=True)
        
        # If the data doesn't have a unit_number column, add a default one
        if 'unit_number' not in df.columns:
            # Check if there's an ID column that could represent units
            id_columns = [col for col in df.columns if any(id_indicator in col.lower() 
                                                         for id_indicator in ['id', 'unit', 'device', 'machine', 'equipment'])]
            
            if id_columns:
                # Use the first identified ID column
                df = df.rename(columns={id_columns[0]: 'unit_number'})
            else:
                # If no ID column found, assume all data is from one unit
                df['unit_number'] = 1
        
        # If we don't have a cycles column, create one based on row index
        if 'time_cycles' not in df.columns:
            if time_columns:
                # If we have a timestamp, calculate elapsed time in appropriate units
                df['time_cycles'] = (df[time_col] - df[time_col].min()).dt.total_seconds() / 3600  # hours
            else:
                # Otherwise use row index as cycle
                df['time_cycles'] = df.index + 1
        
        return df
    
    except Exception as e:
        print(f"Error parsing sensor API data: {str(e)}")
        return None

def _parse_generic_api_data(json_data):
    """Parse any generic JSON API data into a DataFrame"""
    try:
        # Handle different possible JSON structures
        if isinstance(json_data, dict):
            # Look for the actual data array in common JSON API formats
            for key in ['data', 'results', 'items', 'records', 'content']:
                if key in json_data and isinstance(json_data[key], list):
                    return pd.DataFrame(json_data[key])
            
            # If no list found in common fields, convert the dict itself to a single-row DataFrame
            return pd.DataFrame([json_data])
            
        elif isinstance(json_data, list):
            # JSON is already a list, convert directly to DataFrame
            return pd.DataFrame(json_data)
        
        else:
            # Unknown format, try to convert to string and then to single-cell DataFrame
            return pd.DataFrame([{'data': str(json_data)}])
    
    except Exception as e:
        print(f"Error parsing generic API data: {str(e)}")
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
