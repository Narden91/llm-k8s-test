import boto3
import io
from datetime import datetime
from rich.console import Console
from typing import List, Dict


console = Console()


class S3Operations:
    def __init__(self, credentials, bucket_name):
        """Initialize S3 operations with credentials and bucket name"""
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=credentials['aws_endpoint_url'],
            aws_access_key_id=credentials['aws_access_key_id'],
            aws_secret_access_key=credentials['aws_secret_access_key']
        )

    def count_txt_files(self, folder: str) -> int:
        """
        Count the number of .txt files in the specified folder
        
        Args:
            folder: Folder path in the bucket
        
        Returns:
            Integer count of .txt files
        """
        try:
            # Ensure folder path ends with '/'
            folder = folder.rstrip('/') + '/'
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=folder
            )
            
            if 'Contents' not in response:
                return 0
                
            txt_files = [obj for obj in response['Contents'] 
                        if obj['Key'].endswith('.txt')]
            return len(txt_files)
        except Exception as e:
            console.print(f"[red]Error counting txt files: {str(e)}[/]")
            return 0

    def list_image_files(self, folder: str) -> List[str]:
        """
        List all image files in the specified folder
        
        Args:
            folder: Folder path in the bucket
        
        Returns:
            List of image file keys
        """
        try:
            # Ensure folder path ends with '/'
            folder = folder.rstrip('/') + '/'
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=folder
            )
            
            if 'Contents' not in response:
                return []
                
            image_files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            return image_files
        except Exception as e:
            console.print(f"[red]Error listing image files: {str(e)}[/]")
            return []

    def get_image(self, key: str) -> bytes:
        """
        Get image data from S3
        
        Args:
            key: S3 object key
        
        Returns:
            Image data as bytes
        """
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()

    def save_processed_image(self, original_key: str, image_data: bytes, 
                           raw_folder: str, processed_folder: str) -> str:
        """
        Save processed image to processed images folder
        
        Args:
            original_key: Original image key
            image_data: Processed image bytes
            raw_folder: Source folder path
            processed_folder: Destination folder path
        
        Returns:
            S3 URI of saved file
        """
        # Get original filename from the full key
        filename = original_key.split('/')[-1]
        
        # Create new key in processed folder
        new_key = f"{processed_folder.rstrip('/')}/{filename}"
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=new_key,
            Body=image_data
        )
        
        return f"s3://{self.bucket_name}/{new_key}"

    def save_results(self, results: List[Dict], device_info: Dict, folder: str) -> str:
        """
        Save matrix multiplication benchmark results
        
        Args:
            results: List of benchmark results
            device_info: Device information dictionary
            folder: Destination folder for results
        
        Returns:
            S3 URI of saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matrix_multiplication_benchmark_{timestamp}.txt"
        
        # Ensure proper folder path construction
        folder_path = folder.rstrip('/')  # Remove trailing slash if present
        key = f"{folder_path}/{filename}"  # Construct the full path
        
        buffer = io.StringIO()
        
        buffer.write("Matrix Multiplication Benchmark Results\n")
        buffer.write("=====================================\n")
        buffer.write("\nDevice Information:\n")
        for k, value in device_info.items():
            buffer.write(f"{k}: {value}\n")
        
        buffer.write("\nBenchmark Results:\n")
        buffer.write("=====================================\n")
        buffer.write(f"{'Matrix Size':^12} | {'CPU Time (s)':^12} | {'GPU Time (s)':^12} | {'Speedup':^12}\n")
        buffer.write("-" * 55 + "\n")
        
        for result in results:
            gpu_time = f"{result['gpu_time']:.6f}" if result['gpu_time'] is not None else "N/A"
            speedup = f"{result['speedup']:.2f}x" if result['speedup'] is not None else "N/A"
            buffer.write(f"{result['size']:^12} | {result['cpu_time']:^12.6f} | {gpu_time:^12} | {speedup:^12}\n")
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.getvalue()
        )
        
        buffer.close()
        return f"s3://{self.bucket_name}/{key}"

    def save_processing_results(self, results: List[Dict], device_info: Dict, processed_folder: str) -> str:
        """Save image processing benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processing_results_{timestamp}.txt"
        
        # Save in ProcessedImages folder
        key = f"{processed_folder.rstrip('/')}/{filename}"
        
        buffer = io.StringIO()
        buffer.write("Image Processing Results\n")
        buffer.write("=====================\n\n")
        buffer.write("Device Information:\n")
        for k, value in device_info.items():
            buffer.write(f"{k}: {value}\n")
        
        buffer.write("\nProcessing Times:\n")
        buffer.write("─" * 55 + "\n")
        buffer.write(f"{'Image':^12} | {'CPU Time (s)':^12} | {'GPU Time (s)':^12} | {'Speedup':^12}\n")
        buffer.write("─" * 55 + "\n")
        
        for result in results:
            gpu_time = f"{result['gpu_time']:.6f}" if result['gpu_time'] is not None else "N/A"
            speedup = f"{result['speedup']:.2f}x" if result['speedup'] is not None else "N/A"
            buffer.write(f"{result['image_index']:^12} | {result['cpu_time']:^12.6f} | {gpu_time:^12} | {speedup:^12}\n")
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.getvalue()
        )
        
        buffer.close()
        return f"s3://{self.bucket_name}/{key}"