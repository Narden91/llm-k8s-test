import logging
from typing import Dict
import os
from dotenv import load_dotenv

class ConfigHandler:
    def __init__(self):
        self._setup_logging()
        
        # Try to load .env file
        if os.path.exists('.env'):
            load_dotenv()
        
        self.config = {
            'S3_ENDPOINT_URL': os.getenv('S3_ENDPOINT_URL'),
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'S3_BUCKET': os.getenv('S3_BUCKET')
        }
        
        self._validate_config()
    
    def _setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        # Set boto3 and botocore logging to WARNING to reduce noise
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def _validate_config(self) -> None:
        missing = [k for k, v in self.config.items() if v is None]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        logging.info("Config validated successfully!")
    
    def get_aws_credentials(self) -> Dict[str, str]:
        return {
            'aws_endpoint_url': self.config['S3_ENDPOINT_URL'],
            'aws_access_key_id': self.config['AWS_ACCESS_KEY_ID'],
            'aws_secret_access_key': self.config['AWS_SECRET_ACCESS_KEY']
        }
    
    @property
    def s3_bucket(self) -> str:
        return self.config['S3_BUCKET']