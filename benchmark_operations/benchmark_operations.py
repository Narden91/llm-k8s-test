import torch
import numpy as np
import time
from rich.console import Console

console = Console()


class BenchmarkOperations:
    def __init__(self):
        """Initialize benchmark operations with CUDA availability check"""
        self.cuda_available = torch.cuda.is_available()
        self.gpu_device = torch.device("cuda" if self.cuda_available else "cpu")
        self.cpu_device = torch.device("cpu")

    def get_device_info(self):
        """
        Return information about the CUDA device if available
        """
        if self.cuda_available:
            return {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
            }
        return {"device_name": "CPU only - No CUDA device available"}

    def matrix_multiply_gpu(self, matrix_a, matrix_b):
        """
        Multiply two matrices using GPU
        """
        if not self.cuda_available:
            raise RuntimeError("CUDA is not available on this system")
            
        a_tensor = torch.tensor(matrix_a, dtype=torch.float32).to(self.gpu_device)
        b_tensor = torch.tensor(matrix_b, dtype=torch.float32).to(self.gpu_device)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        result = torch.mm(a_tensor, b_tensor)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        result = result.cpu().numpy()
        return result, end_time - start_time

    def matrix_multiply_cpu(self, matrix_a, matrix_b):
        """
        Multiply two matrices using CPU
        """
        a_tensor = torch.tensor(matrix_a, dtype=torch.float32)
        b_tensor = torch.tensor(matrix_b, dtype=torch.float32)
        
        start_time = time.perf_counter()
        result = torch.mm(a_tensor, b_tensor)
        end_time = time.perf_counter()
        
        return result.numpy(), end_time - start_time

    def run_comparison(self, sizes=[1000, 2000, 3000, 4000]):
        """
        Run comparison tests for different matrix sizes
        """
        results = []
        device_info = self.get_device_info()
        
        for size in sizes:
            console.print(f"\nTesting {size}x{size} matrices...")
            
            matrix_a = np.random.rand(size, size)
            matrix_b = np.random.rand(size, size)
            
            console.print("Running CPU multiplication...")
            _, cpu_time = self.matrix_multiply_cpu(matrix_a, matrix_b)
            
            gpu_time = None
            if self.cuda_available:
                console.print("Running GPU multiplication...")
                _, gpu_time = self.matrix_multiply_gpu(matrix_a, matrix_b)
                torch.cuda.empty_cache()
            
            results.append({
                'size': size,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': (cpu_time / gpu_time) if gpu_time and gpu_time > 0 else None
            })
        
        return results, device_info