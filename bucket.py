import boto3
from botocore.exceptions import ClientError
from dotenv import dotenv_values

# CONFIG
BUCKET_NAME = 'transformerwildfire'
S3_ENDPOINT = 'https://fsn1.your-objectstorage.com'

def check_bucket():
    config = dotenv_values(".env")
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=config.get('ACCESS_KEY'),
        aws_secret_access_key=config.get('SECRET_KEY'),
    )

    print(f"--- Probing bucket: {BUCKET_NAME} ---")

    try:
        # head_bucket returns a 200 OK if the bucket exists and you have access
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"✅ Success: Bucket '{BUCKET_NAME}' exists and is accessible.")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code == '404':
            print(f"❌ Error: Bucket '{BUCKET_NAME}' does not exist (404).")
        elif error_code == '403':
            print(f"❌ Error: Bucket exists, but Access Denied (403). Check your credentials.")
        else:
            print(f"❌ Error: Received unexpected status code {error_code}.")
            print(f"Full error: {e}")
            
    except Exception as e:
        print(f"❌ Connection Error: Could not reach the endpoint at {S3_ENDPOINT}.")
        print(f"Details: {e}")

if __name__ == "__main__":
    check_bucket()