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
        config=Config(signature_version=UNSIGNED) # This tells AWS "I am anonymous"
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

def main():
    parser = argparse.ArgumentParser(
        description="Download a single file from S3"
    )

    parser.add_argument("--s3_path", type=str, required=True, help="The full S3 key/path of the file to download (e.g., data/raw/dataset.pt)")
    parser.add_argument("--local_dest", type=Path, required=True, help="Path to the local destination folder")

    args = parser.parse_args()

    download_single_file(args.s3_path, args.local_dest)

if __name__ == '__main__':
    main()