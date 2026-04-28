import argparse
import boto3
import os
from pathlib import Path
from dotenv import dotenv_values

# CONFIG
BUCKET_NAME = 'transformerwildfire'
S3_ENDPOINT = f'https://fsn1.your-objectstorage.com'

def get_s3_client():
    """Helper to initialize the S3 client."""
    config = dotenv_values(".env")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )

def upload_single_file(s3_client, local_file_path, s3_prefix):
    """Uploads a single file to S3 under the specified prefix."""
    file_path = Path(local_file_path)
    s3_key = os.path.join(s3_prefix, file_path.name).replace("\\", "/")

    print(f"--- Uploading File: {file_path.name} -> s3://{BUCKET_NAME}/{s3_key} ---")

    try:
        s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
        print("--- File Upload Complete ---")
    except Exception as e:
        print(f"Failed to upload {file_path.name}: {e}")

def upload_directory(s3_client, local_directory_path, s3_prefix):
    """
    Uploads the entire folder (and its contents) to S3.
    """
    dir_path = Path(local_directory_path).resolve()
    # To include the folder itself, we calculate relative to the parent
    parent_path = dir_path.parent
    
    print(f"--- Uploading Directory: {dir_path.name} -> s3://{BUCKET_NAME}/{s3_prefix}/{dir_path.name} ---")

    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            # relative_to(parent_path) ensures the folder name is included in the path
            relative_path = file_path.relative_to(parent_path)
            s3_key = os.path.join(s3_prefix, str(relative_path)).replace("\\", "/")
            print(f'S3 Key:', s3_key)
            try:
                s3_client.upload_file(str(file_path), BUCKET_NAME, s3_key)
                print(f"  Uploaded: {relative_path}")
            except Exception as e:
                print(f"  Failed to upload {file_path}: {e}")
    
    print("--- Directory Upload Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Upload file or folder to S3")
    parser.add_argument("--path", type=Path, required=True, help="Local path to file or directory")
    parser.add_argument("--s3_prefix", type=str, required=True, help="S3 destination prefix")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path '{args.path}' does not exist.")
        return

    client = get_s3_client()

    if args.path.is_file():
        upload_single_file(client, args.path, args.s3_prefix)
    elif args.path.is_dir():
        upload_directory(client, args.path, args.s3_prefix)

if __name__ == '__main__':
    main()