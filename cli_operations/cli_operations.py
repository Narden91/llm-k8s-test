import argparse
import os
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv


console = Console()


class CLIOperations:
    def __init__(self):
        load_dotenv()
    
    @staticmethod
    def get_processing_mode():
        """
        Get processing mode from command line or environment variable
        Returns:
            Tuple of (mode, source)
        """
        parser = argparse.ArgumentParser(description='GPU Processing Benchmark')
        parser.add_argument(
            '--mode',
            type=str,
            choices=['matrix', 'image'],
            help='Processing mode: matrix or image'
        )
        args = parser.parse_args()
        
        # Check command line arguments first
        if args.mode is not None:
            return args.mode, "command line"
        
        # Check environment variable
        env_mode = os.getenv('PROCESSING_MODE')
        if env_mode and env_mode.lower() in ['matrix', 'image']:
            return env_mode.lower(), "environment variable"
        
        # Use default value if neither is provided
        return "matrix", "default value"
    
    @staticmethod
    def display_image_results(results):
        """Display image processing benchmark results"""
        console.print("\n[bold]Image Processing Summary:[/]")
        console.print("─" * 55)
        console.print(f"{'Image':^12} | {'CPU Time (s)':^12} | {'GPU Time (s)':^12} | {'Speedup':^12}")
        console.print("─" * 55)
        
        for result in results:
            gpu_time = f"{result['gpu_time']:.6f}" if result['gpu_time'] is not None else "N/A"
            speedup = f"{result['speedup']:.2f}x" if result['speedup'] is not None else "N/A"
            
            speedup_color = "yellow"
            if result['speedup']:
                if result['speedup'] > 2:
                    speedup_color = "bright_green"
                elif result['speedup'] > 1:
                    speedup_color = "green"
            
            console.print(
                f"{result['image_index']:^12} | "
                f"{result['cpu_time']:^12.6f} | "
                f"{gpu_time:^12} | "
                f"[{speedup_color}]{speedup:^12}[/]"
            )
    
    @staticmethod
    def parse_matrix_sizes(sizes_str):
        """
        Parse matrix sizes string into a list of integers
        """
        try:
            return [int(size.strip()) for size in sizes_str.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Matrix sizes must be comma-separated integers")

    @staticmethod
    def get_matrix_sizes():
        """
        Get matrix sizes from command line or environment variable
        """
        parser = argparse.ArgumentParser(description='Matrix multiplication benchmark')
        parser.add_argument(
            '--matrix-sizes',
            type=CLIOperations.parse_matrix_sizes,
            help='Comma-separated list of matrix sizes (e.g., 1000,2000,3000)'
        )
        args = parser.parse_args()
        
        # Check command line arguments first
        if args.matrix_sizes is not None:
            return args.matrix_sizes, "command line"
        
        # Check environment variable
        env_sizes = os.getenv('MATRIX_SIZES')
        if env_sizes:
            return CLIOperations.parse_matrix_sizes(env_sizes), "environment variable"
        
        # Use default values if neither is provided
        default_sizes = [1000, 2000, 3000]
        return default_sizes, "default values"

    @staticmethod
    def display_configuration(sizes, source):
        """Display benchmark configuration"""
        console.print(Panel(
            f"[bold green]Matrix Multiplication Benchmark Configuration[/]\n\n"
            f"[yellow]Matrix sizes source:[/] {source}\n"
            f"[yellow]Matrix sizes:[/] {', '.join(map(str, sizes))}",
            title="Configuration",
            style="blue"
        ))

    @staticmethod
    def display_results(results):
        """Display benchmark results"""
        console.print("\n[bold]Summary:[/]")
        console.print("─" * 55)
        console.print(f"{'Matrix Size':^12} | {'CPU Time (s)':^12} | {'GPU Time (s)':^12} | {'Speedup':^12}")
        console.print("─" * 55)
        
        for result in results:
            gpu_time = f"{result['gpu_time']:.6f}" if result['gpu_time'] is not None else "N/A"
            speedup = f"{result['speedup']:.2f}x" if result['speedup'] is not None else "N/A"
            
            speedup_color = "yellow"
            if result['speedup']:
                if result['speedup'] > 2:
                    speedup_color = "bright_green"
                elif result['speedup'] > 1:
                    speedup_color = "green"
                
            console.print(
                f"{result['size']:^12} | "
                f"{result['cpu_time']:^12.6f} | "
                f"{gpu_time:^12} | "
                f"[{speedup_color}]{speedup:^12}[/]"
            )