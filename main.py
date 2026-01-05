import sys
sys.dont_write_bytecode = True

from config.s3_config_handler import ConfigHandler
from s3_operations.s3_operations import S3Operations
from benchmark_operations.benchmark_operations import BenchmarkOperations
from image_processing.image_processing_operations import ImageProcessingOperations
from cli_operations.cli_operations import CLIOperations
from rich.console import Console
from rich.panel import Panel
import argparse
import os

console = Console()

def get_folder_paths():
    """Get folder paths from command line, environment, or defaults"""
    parser = argparse.ArgumentParser(description='GPU Processing Benchmark')
    parser.add_argument('--raw-images-folder', type=str, help='Raw images folder path in S3')
    parser.add_argument('--processed-images-folder', type=str, help='Processed images folder path in S3')
    parser.add_argument('--results-folder', type=str, help='Benchmark results folder path in S3')
    args = parser.parse_args()

    # Command line arguments take precedence
    raw_folder = args.raw_images_folder
    processed_folder = args.processed_images_folder
    results_folder = args.results_folder

    # Check environment variables if not in command line
    if not raw_folder:
        raw_folder = os.getenv('RAW_IMAGES_FOLDER', 'RawImages')
    if not processed_folder:
        processed_folder = os.getenv('PROCESSED_IMAGES_FOLDER', 'ProcessedImages')
    if not results_folder:
        results_folder = os.getenv('RESULTS_FOLDER', 'benchmark_results')

    return raw_folder, processed_folder, results_folder

def process_matrices(s3_ops, cli_ops, results_folder):
    """Handle matrix multiplication benchmark"""
    sizes, source = cli_ops.get_matrix_sizes()
    cli_ops.display_configuration(sizes, source)
    
    txt_count = s3_ops.count_txt_files(results_folder)
    console.print(f"\n[cyan]Number of existing benchmark files:[/] {txt_count}")
    
    benchmark_ops = BenchmarkOperations()
    console.print("\n[bold cyan]Starting matrix multiplication benchmark...[/]")
    results, device_info = benchmark_ops.run_comparison(sizes)
    
    filename = s3_ops.save_results(results, device_info, results_folder)
    console.print(f"\n[green]Benchmark complete! Results saved to:[/] {filename}")
    
    cli_ops.display_results(results)

def process_images(s3_ops, cli_ops, raw_folder, processed_folder, results_folder):
    """Handle image processing benchmark"""
    # List images in raw images folder
    image_files = s3_ops.list_image_files(raw_folder)
    if not image_files:
        console.print(f"[red]No images found in {raw_folder} folder[/]")
        return
    
    console.print(f"\nFound {len(image_files)} images to process")
    
    # Initialize image processing operations
    image_ops = ImageProcessingOperations()
    
    # Process all images
    image_data = []
    for image_file in image_files:
        try:
            data = s3_ops.get_image(image_file)
            image_data.append(data)
        except Exception as e:
            console.print(f"[red]Error loading image {image_file}: {str(e)}[/]")
    
    console.print("\n[bold cyan]Starting image processing benchmark...[/]")
    results, device_info = image_ops.process_batch(image_data)
    
    # Save processed images and results
    for image_file, result in zip(image_files, results):
        try:
            s3_uri = s3_ops.save_processed_image(
                image_file, 
                result['processed_data'],
                raw_folder,
                processed_folder
            )
            console.print(f"[green]Saved processed image to:[/] {s3_uri}")
        except Exception as e:
            console.print(f"[red]Error saving processed image {image_file}: {str(e)}[/]")
    
    # Save benchmark results in ProcessedImages folder
    results_uri = s3_ops.save_processing_results(results, device_info, processed_folder)
    console.print(f"\n[green]Benchmark results saved to:[/] {results_uri}")
    
    cli_ops.display_image_results(results)

def main():
    """Main function to run the GPU processing benchmark"""
    # Get folder paths
    raw_folder, processed_folder, results_folder = get_folder_paths()
    
    # Initialize operations
    cli_ops = CLIOperations()
    mode, source = cli_ops.get_processing_mode()
    
    console.print(Panel(
        f"[bold green]GPU Processing Benchmark Configuration[/]\n\n"
        f"[yellow]Processing mode:[/] {mode} (from {source})\n"
        f"[yellow]Raw images folder:[/] {raw_folder}\n"
        f"[yellow]Processed images folder:[/] {processed_folder}\n"
        f"[yellow]Results folder:[/] {results_folder}",
        title="Configuration",
        style="blue"
    ))
    
    
    print("\n")
    
    # Initialize configuration handler for S3 access
    config = ConfigHandler()
    
    # Initialize S3 operations
    s3_ops = S3Operations(config.get_aws_credentials(), config.s3_bucket)
    
    # Run appropriate processing mode
    if mode == "matrix":
        process_matrices(s3_ops, cli_ops, results_folder)
    else:  # mode == "image"
        process_images(s3_ops, cli_ops, raw_folder, processed_folder, results_folder)

if __name__ == "__main__":
    main()