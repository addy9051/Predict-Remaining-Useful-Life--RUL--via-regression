import boto3
import os
import pandas as pd
import io
from botocore.exceptions import ClientError, NoCredentialsError

def check_aws_connection():
    """
    Check if we can connect to AWS using the provided credentials
    
    Returns:
        Boolean indicating successful connection
    """
    try:
        # Try to connect to S3
        s3_client = boto3.client('s3')
        s3_client.list_buckets()
        return True
    except (ClientError, NoCredentialsError):
        return False

def list_s3_buckets():
    """
    List all S3 buckets accessible with current credentials
    
    Returns:
        List of bucket names or None if error
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        return buckets
    except Exception as e:
        print(f"Error listing S3 buckets: {str(e)}")
        return None

def list_s3_objects(bucket_name, prefix=''):
    """
    List objects in a specific S3 bucket
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Prefix to filter objects (folder-like)
        
    Returns:
        List of object keys or None if error
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []
    except Exception as e:
        print(f"Error listing objects in bucket {bucket_name}: {str(e)}")
        return None

def upload_to_s3(file_path, bucket_name, object_key):
    """
    Upload a file to S3
    
    Args:
        file_path: Path to the local file
        bucket_name: Target S3 bucket
        object_key: S3 object key (path in bucket)
        
    Returns:
        Boolean indicating success
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, bucket_name, object_key)
        return True
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        return False

def download_from_s3(bucket_name, object_key, file_path):
    """
    Download a file from S3
    
    Args:
        bucket_name: Source S3 bucket
        object_key: S3 object key to download
        file_path: Local path to save the file
        
    Returns:
        Boolean indicating success
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, file_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {str(e)}")
        return False

def read_csv_from_s3(bucket_name, object_key):
    """
    Read a CSV file directly from S3 into a pandas DataFrame
    
    Args:
        bucket_name: Source S3 bucket
        object_key: S3 object key of the CSV file
        
    Returns:
        pandas DataFrame or None if error
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        return pd.read_csv(io.BytesIO(response['Body'].read()))
    except Exception as e:
        print(f"Error reading CSV from S3: {str(e)}")
        return None

def create_s3_bucket(bucket_name, region=None):
    """
    Create a new S3 bucket
    
    Args:
        bucket_name: Name for the new bucket
        region: AWS region (optional)
        
    Returns:
        Boolean indicating success
    """
    try:
        s3_client = boto3.client('s3')
        if region is None:
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration=location
            )
        return True
    except Exception as e:
        print(f"Error creating S3 bucket: {str(e)}")
        return False

def delete_s3_object(bucket_name, object_key):
    """
    Delete an object from S3
    
    Args:
        bucket_name: S3 bucket name
        object_key: Key of object to delete
        
    Returns:
        Boolean indicating success
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
        return True
    except Exception as e:
        print(f"Error deleting object from S3: {str(e)}")
        return False
