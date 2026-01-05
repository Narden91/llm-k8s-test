from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import time
from rich.console import Console
from typing import Tuple, List, Dict

console = Console()

class ImageProcessingOperations:
    def __init__(self):
        """Initialize image processing operations with CUDA availability check"""
        self.cuda_available = torch.cuda.is_available()
        self.gpu_device = torch.device("cuda" if self.cuda_available else "cpu")
        self.cpu_device = torch.device("cpu")
        
        # Create transform pipeline
        self.blur = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(31, 31), sigma=5.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        ])
        
    def get_device_info(self):
        """Return information about the CUDA device if available"""
        if self.cuda_available:
            return {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
            }
        return {"device_name": "CPU only - No CUDA device available"}

    def process_image_gpu(self, image_data: bytes) -> Tuple[bytes, float]:
        """
        Process image using GPU
        Args:
            image_data: Raw image bytes
        Returns:
            Tuple of (processed image bytes, processing time)
        """
        if not self.cuda_available:
            raise RuntimeError("CUDA is not available on this system")

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to tensor and move to GPU
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.gpu_device)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Apply blur on GPU
        processed_tensor = self.blur(image_tensor)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Convert back to image
        processed_tensor = processed_tensor.cpu().squeeze(0)
        processed_image = transforms.ToPILImage()(processed_tensor)
        
        # Convert to bytes
        buffer = io.BytesIO()
        processed_image.save(buffer, format=image.format if image.format else 'PNG')
        
        return buffer.getvalue(), end_time - start_time

    def process_image_cpu(self, image_data: bytes) -> Tuple[bytes, float]:
        """
        Process image using CPU
        Args:
            image_data: Raw image bytes
        Returns:
            Tuple of (processed image bytes, processing time)
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        start_time = time.perf_counter()
        
        # Apply blur on CPU
        processed_tensor = self.blur(image_tensor)
        
        end_time = time.perf_counter()
        
        # Convert back to image
        processed_tensor = processed_tensor.squeeze(0)
        processed_image = transforms.ToPILImage()(processed_tensor)
        
        # Convert to bytes
        buffer = io.BytesIO()
        processed_image.save(buffer, format=image.format if image.format else 'PNG')
        
        return buffer.getvalue(), end_time - start_time

    def process_batch(self, images_data: List[bytes]) -> Tuple[List[Dict], Dict]:
        """
        Process a batch of images and compare CPU vs GPU performance
        Args:
            images_data: List of image bytes
        Returns:
            Tuple of (results list, device info)
        """
        results = []
        device_info = self.get_device_info()
        
        for idx, image_data in enumerate(images_data):
            console.print(f"\nProcessing image {idx + 1}/{len(images_data)}...")
            
            console.print("Processing on CPU...")
            processed_cpu, cpu_time = self.process_image_cpu(image_data)
            
            gpu_time = None
            processed_gpu = None
            if self.cuda_available:
                console.print("Processing on GPU...")
                processed_gpu, gpu_time = self.process_image_gpu(image_data)
                torch.cuda.empty_cache()
            
            results.append({
                'image_index': idx,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': (cpu_time / gpu_time) if gpu_time and gpu_time > 0 else None,
                'processed_data': processed_gpu if processed_gpu else processed_cpu
            })
        
        return results, device_info