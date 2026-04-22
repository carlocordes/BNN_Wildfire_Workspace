import argparse
import boto3
import os
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import dotenv_values

# CONFIG
BUCKET_NAME = 'transformerwildfire'
S3_ENDPOINT = f'https://fsn1.your-objectstorage.com'

def upload_single_file(local_file_path, s3_prefix):
    """
    Uploads a single file to S3 under the specified prefix.
    """
    # Load credentials from .env
    config = dotenv_values(".env")

    # Initialize the S3 client
    # Note: Fixed the typo 'ACCES_KEY' to 'ACCESS_KEY' to match standard naming
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )

    file_path = Path(local_file_path)
    
    # Construct the S3 key (Prefix + filename)
    # This places the file directly inside the prefix folder
    s3_key = os.path.join(s3_prefix, file_path.name).replace("\\", "/")

    print(f"--- Starting upload: {file_path.name} -> s3://{BUCKET_NAME}/{s3_key} ---")

    try:
        s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
        print("--- Upload Complete ---")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Failed to upload {file_path.name}: {e}")


def upload_directory(local_directory_path, s3_prefix):
    """
    Uploads all contents of a directory to S3, preserving the folder structure.
    """
    config = dotenv_values(".env")
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )

    dir_path = Path(local_directory_path)
    print(f"--- Starting directory upload: {dir_path} -> s3://{BUCKET_NAME}/{s3_prefix} ---")

    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            # Calculate relative path to maintain structure inside the S3 prefix
            relative_path = file_path.relative_to(dir_path)
            s3_key = os.path.join(s3_prefix, str(relative_path)).replace("\\", "/")
            
            try:
                s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
                print(f"Uploaded: {relative_path}")
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")
    
    print("--- Directory Upload Complete ---")

def main():
    parser = argparse.ArgumentParser(
        description="Upload a single file to S3"
    )

    # Changed from --local_folder to --file_path
    parser.add_argument("--file_path", type=Path, required=True, help="Path to the local file")
    parser.add_argument("--s3_prefix", type=str, required=True, help="S3 folder (prefix) where the file will live")

    args = parser.parse_args()

    # Verify it is a file and not a directory
    if not args.file_path.is_file():
        print(f"Error: {args.file_path} is not a valid file.")
        return

    upload_single_file(args.file_path, args.s3_prefix)

if __name__ == '__main__':
    main()