import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError
import time
import numpy as np
from typing import Any, Dict
import logging


class S3Client:
    def __init__(self, credentials: Dict[str, str], max_retries: int = 5):
        self.credentials = credentials
        self.max_retries = max_retries
        self.client = self._initialize_client()
        self.logger = logging.getLogger(__name__)

    def _initialize_client(self):
        config = Config(
            connect_timeout=30,
            read_timeout=60,
            retries={'max_attempts': 5, 'mode': 'adaptive'},
            max_pool_connections=50,
            tcp_keepalive=True
        )
        return boto3.client(
            's3',
            endpoint_url=self.credentials['aws_endpoint_url'],
            aws_access_key_id=self.credentials['aws_access_key_id'],
            aws_secret_access_key=self.credentials['aws_secret_access_key'],
            config=config
        )

    def _exponential_backoff(self, attempt: int) -> float:
        base_delay = min(300, 2 ** attempt)
        jitter = np.random.uniform(0, 0.1 * base_delay)
        return base_delay + jitter