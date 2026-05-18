import argparse
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import dotenv_values

# CONFIG
BUCKET_NAME = 'transformerwildfire'
S3_ENDPOINT = 'https://fsn1.your-objectstorage.com'

def download_single_file(s3_path, local_folder):
    """
    Downloads a single file from S3 to the specified local folder.
    """
    # Load credentials from .env
    config = dotenv_values(".env")

    # Initialize the S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )

    local_dir = Path(local_folder)
    
    # Create the local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract the filename from the S3 path and construct the full local path
    filename = s3_path.split("/")[-1]
    local_file_path = local_dir / filename

    print(f"--- Starting download: s3://{BUCKET_NAME}/{s3_path} -> {local_file_path} ---")

    try:
        s3_client.download_file(BUCKET_NAME, s3_path, str(local_file_path))
        print("--- Download Complete ---")
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Error: The file {s3_path} was not found in the bucket.")
        else:
            print(f"AWS Client Error: {e}")
    except NoCredentialsError:
        print("Error: AWS credentials not found. Check your .env file.")
    except Exception as e:
        print(f"Failed to download {s3_path}: {e}")

def download_s3_directory(s3_prefix, local_folder):
    """
    Downloads the entire S3 directory (including the target folder itself) 
    and its contents to the local folder.
    """

    config = dotenv_values(".env")

    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )
    # Clean up prefix: remove leading slash, ensure trailing slash
    s3_prefix = s3_prefix.strip('/')
    if s3_prefix:
        s3_prefix += '/'

    local_dir = Path(local_folder)
    print(f"--- Scanning s3://{BUCKET_NAME}/{s3_prefix} for files ---")

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix)

        file_count = 0
        
        # Calculate the parent prefix path so we keep the target folder name
        # e.g., if s3_prefix is "data/raw/", prefix_parent is "data/"
        prefix_path = Path(s3_prefix)
        prefix_parent = prefix_path.parent if len(prefix_path.parts) > 1 else ""

        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                if s3_key.endswith('/'):
                    continue

                # NEW LOGIC: Calculate relative path from the PARENT of the prefix
                # e.g., s3_key = "data/raw/nested/file.txt", prefix_parent = "data"
                # relative_path = "raw/nested/file.txt" (keeps the "raw" folder)
                if prefix_parent:
                    relative_path = Path(s3_key).relative_to(prefix_parent)
                else:
                    relative_path = Path(s3_key)

                local_file_path = local_dir / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"Downloading: {s3_key} -> {local_file_path}")
                s3_client.download_file(BUCKET_NAME, s3_key, str(local_file_path))
                file_count += 1

        if file_count == 0:
            print(f"No files found matching prefix: {s3_prefix}")
        else:
            print(f"--- Download Complete. Total files: {file_count} ---")

    except ClientError as e:
        print(f"AWS Client Error: {e}")
    except NoCredentialsError:
        print("Error: AWS credentials not found. Check your .env file.")
    except Exception as e:
        print(f"Failed to download directory: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Download tools for Object Storage / S3"
    )
    
    # Create subparsers for the different modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Download target mode: file or directory")

    # 'file' subcommand setup
    file_parser = subparsers.add_parser("file", help="Download a single file")
    file_parser.add_argument("--s3_path", type=str, required=True, help="The full S3 key/path of the file to download")
    file_parser.add_argument("--local_dest", type=Path, required=True, help="Path to the local destination folder")

    # 'dir' subcommand setup
    dir_parser = subparsers.add_parser("dir", help="Download a directory recursively")
    dir_parser.add_argument("--s3_path", type=str, required=True, help="The S3 prefix/directory to download")
    dir_parser.add_argument("--local_dest", type=Path, required=True, help="Path to the local destination folder")

    args = parser.parse_args()

    # Route execution based on the chosen mode
    if args.mode == "file":
        download_single_file(args.s3_path, args.local_dest)
    elif args.mode == "dir":
        download_s3_directory(args.s3_path, args.local_dest)

if __name__ == '__main__':
    main()