from config.s3_config_handler import ConfigHandler
from s3_operations.s3_operations import S3Operations
import sys

def verify_s3():
    try:
        print("Loading configuration...")
        config = ConfigHandler()
        credentials = config.get_aws_credentials()
        bucket_name = config.s3_bucket
        print(f"Configuration loaded. Bucket: {bucket_name}")
        
        print("Initializing S3 Operations...")
        s3_ops = S3Operations(credentials, bucket_name)
        
        print("Testing connection by listing files...")
        response = s3_ops.s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
        
        if 'Contents' in response:
            print(f"Successfully listed {len(response['Contents'])} items.")
            for obj in response['Contents']:
                print(f" - {obj['Key']}")
        else:
            print("Bucket is empty or no permissions to list (or folder is empty).")
            
        print("S3 Connection verified!")
        return True
    except Exception as e:
        print(f"Failed to verify S3 connection: {e}")
        return False

if __name__ == "__main__":
    success = verify_s3()
    sys.exit(0 if success else 1)
