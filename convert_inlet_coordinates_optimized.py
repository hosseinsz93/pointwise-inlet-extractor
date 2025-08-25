#!/usr/bin/env python3
"""
Highly optimized parallel version with additional performance improvements:
- Memory mapping for large files
- Vectorized operations
- Optimized I/O
- Progress tracking
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import os
import mmap
from tqdm import tqdm
import psutil

def get_optimal_chunk_size(file_size, num_workers):
    """
    Calculate optimal chunk size based on file size and available memory
    """
    available_memory = psutil.virtual_memory().available
    # Use 50% of available memory for processing
    memory_per_worker = (available_memory * 0.5) / num_workers
    
    # Estimate memory needed per line (approximately 50 bytes)
    lines_per_chunk = int(memory_per_worker / 50)
    
    # Ensure reasonable bounds
    min_chunk = 1000
    max_chunk = 50000
    
    chunk_size = max(min_chunk, min(max_chunk, lines_per_chunk))
    return chunk_size

def parse_coordinate_chunk_vectorized(chunk_data):
    """
    Vectorized parsing of coordinate chunks using numpy
    
    Parameters:
    chunk_data (tuple): (chunk_lines, chunk_index)
    
    Returns:
    tuple: (coordinates_array, chunk_index)
    """
    lines, chunk_idx = chunk_data
    
    # Filter out comments and empty lines
    valid_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            valid_lines.append(line)
    
    if not valid_lines:
        return np.array([]), chunk_idx
    
    try:
        # Use numpy's fromstring for faster parsing
        # Join lines and use numpy's built-in parsing
        data_str = '\n'.join(valid_lines)
        
        # Convert to numpy array directly
        coords = np.fromstring(data_str, sep=' ').reshape(-1, 3)
        return coords, chunk_idx
        
    except Exception as e:
        # Fallback to manual parsing if numpy fails
        coordinates = []
        for line in valid_lines:
            parts = line.split()
            if len(parts) == 3:
                try:
                    coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                    coordinates.append(coords)
                except ValueError:
                    continue
        
        return np.array(coordinates), chunk_idx

def read_file_with_mmap(filename, chunk_size):
    """
    Read file using memory mapping for better performance
    
    Parameters:
    filename (str): Input file path
    chunk_size (int): Lines per chunk
    
    Returns:
    list: List of line chunks with indices
    """
    chunks = []
    
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            current_chunk = []
            chunk_idx = 0
            
            for line in iter(mmapped_file.readline, b""):
                current_chunk.append(line.decode('utf-8'))
                
                if len(current_chunk) >= chunk_size:
                    chunks.append((current_chunk, chunk_idx))
                    current_chunk = []
                    chunk_idx += 1
            
            # Add remaining lines
            if current_chunk:
                chunks.append((current_chunk, chunk_idx))
    
    return chunks

def read_inlet_coordinates_optimized(input_file, output_file, num_workers=None, 
                                   use_progress_bar=True):
    """
    Highly optimized parallel coordinate reading with memory mapping and vectorization
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 for most efficient processing
    
    print(f"Optimized parallel processing with {num_workers} workers")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Get file size for optimization
    file_size = os.path.getsize(input_file)
    print(f"File size: {file_size / (1024**2):.1f} MB")
    
    # Calculate optimal chunk size
    chunk_size = get_optimal_chunk_size(file_size, num_workers)
    print(f"Using chunk size: {chunk_size} lines")
    
    start_time = time.time()
    
    # Read file with memory mapping
    print("Reading file with memory mapping...")
    chunks = read_file_with_mmap(input_file, chunk_size)
    print(f"Split into {len(chunks)} chunks")
    
    # Process chunks in parallel with progress bar
    print("Processing chunks in parallel...")
    
    all_coordinates = [None] * len(chunks)  # Pre-allocate for ordered results
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(parse_coordinate_chunk_vectorized, chunk_data): i 
                          for i, chunk_data in enumerate(chunks)}
        
        # Use progress bar if requested
        if use_progress_bar:
            progress_bar = tqdm(total=len(chunks), desc="Processing chunks")
        
        # Collect results
        for future in as_completed(future_to_chunk):
            chunk_original_idx = future_to_chunk[future]
            try:
                coords, chunk_idx = future.result()
                all_coordinates[chunk_idx] = coords
                
                if use_progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'points': len(coords) if len(coords.shape) > 1 else 0
                    })
                
            except Exception as exc:
                print(f"Chunk {chunk_original_idx} generated an exception: {exc}")
                all_coordinates[chunk_idx] = np.array([])
        
        if use_progress_bar:
            progress_bar.close()
    
    # Concatenate all coordinate arrays
    print("Combining results...")
    valid_coords = [coords for coords in all_coordinates if coords.size > 0]
    
    if valid_coords:
        coords_array = np.vstack(valid_coords)
    else:
        coords_array = np.array([])
    
    print(f"Total points processed: {len(coords_array)}")
    
    if len(coords_array) == 0:
        print("Warning: No valid coordinates found!")
        return np.array([]), np.array([]), np.array([])
    
    # Extract coordinate arrays
    x_coords = coords_array[:, 0]
    y_coords = coords_array[:, 1]
    z_coords = coords_array[:, 2]
    
    read_time = time.time() - start_time
    print(f"Reading and parsing completed in {read_time:.2f} seconds")
    print(f"Processing rate: {len(coords_array) / read_time:.0f} points/second")
    
    # Write output file efficiently
    print(f"Writing to {output_file}...")
    write_start = time.time()
    
    with open(output_file, 'w') as f:
        f.write(f"# Inlet boundary coordinates converted from {input_file}\n")
        f.write(f"# Total points: {len(coords_array)}\n")
        f.write(f"# Format: X Y Z coordinates\n")
        f.write(f"# Processed with {num_workers} workers in {read_time:.2f}s\n")
        f.write("\n")
        
        # Use numpy's efficient writing
        np.savetxt(f, coords_array, fmt='%.6f', delimiter='\t')
    
    write_time = time.time() - write_start
    print(f"Writing completed in {write_time:.2f} seconds")
    
    # Statistics
    print(f"\nCoordinate Statistics:")
    print(f"X: min={x_coords.min():.6f}, max={x_coords.max():.6f}, std={x_coords.std():.6f}")
    print(f"Y: min={y_coords.min():.6f}, max={y_coords.max():.6f}, std={y_coords.std():.6f}")
    print(f"Z: min={z_coords.min():.6f}, max={z_coords.max():.6f}, std={z_coords.std():.6f}")
    
    return x_coords, y_coords, z_coords

def save_arrays_async(x_coords, y_coords, z_coords, prefix="inlet_coords_optimized"):
    """
    Asynchronous saving of coordinate arrays
    """
    print(f"\nSaving arrays asynchronously...")
    start_time = time.time()
    
    def save_single_array(args):
        coords, filename, format_type = args
        if format_type == 'text':
            np.savetxt(filename, coords, fmt='%.6f')
        elif format_type == 'binary':
            np.save(filename, coords)
        return filename
    
    # Prepare all save tasks
    save_tasks = [
        (x_coords, f"{prefix}_x.txt", 'text'),
        (y_coords, f"{prefix}_y.txt", 'text'),
        (z_coords, f"{prefix}_z.txt", 'text'),
        (x_coords, f"{prefix}_x.npy", 'binary'),
        (y_coords, f"{prefix}_y.npy", 'binary'),
        (z_coords, f"{prefix}_z.npy", 'binary'),
    ]
    
    # Use ThreadPoolExecutor for I/O operations
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(save_single_array, task) for task in save_tasks]
        
        # Use progress bar for saving
        with tqdm(total=len(futures), desc="Saving files") as pbar:
            for future in as_completed(futures):
                filename = future.result()
                pbar.update(1)
                pbar.set_postfix({'file': os.path.basename(filename)})
    
    save_time = time.time() - start_time
    print(f"All files saved in {save_time:.2f} seconds")
    
    # Show file sizes
    print(f"\nGenerated files:")
    for coords, base_name, _ in [(x_coords, 'x', None), (y_coords, 'y', None), (z_coords, 'z', None)]:
        txt_file = f"{prefix}_{base_name}.txt"
        npy_file = f"{prefix}_{base_name}.npy"
        
        txt_size = os.path.getsize(txt_file) / (1024**2)
        npy_size = os.path.getsize(npy_file) / (1024**2)
        
        print(f"{base_name.upper()}: {txt_file} ({txt_size:.1f} MB), {npy_file} ({npy_size:.1f} MB)")

if __name__ == "__main__":
    input_file = "inlet_boundary_coordinates.txt"
    output_file = "xyz_optimized.dat"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        exit(1)
    
    # System information
    print("=" * 60)
    print("OPTIMIZED PARALLEL COORDINATE CONVERTER")
    print("=" * 60)
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Get user preferences
    try:
        num_workers = int(input(f"Number of workers (1-{mp.cpu_count()}, default={min(mp.cpu_count(), 8)}): ") or min(mp.cpu_count(), 8))
        num_workers = max(1, min(num_workers, mp.cpu_count()))
    except ValueError:
        num_workers = min(mp.cpu_count(), 8)
    
    use_progress = input("Show progress bars? (Y/n): ").lower().strip() != 'n'
    
    print(f"\nStarting optimized processing with {num_workers} workers...")
    
    total_start = time.time()
    
    try:
        # Process coordinates
        x_coords, y_coords, z_coords = read_inlet_coordinates_optimized(
            input_file, output_file, num_workers, use_progress)
        
        if len(x_coords) > 0:
            # Save arrays
            save_arrays_async(x_coords, y_coords, z_coords)
            
            total_time = time.time() - total_start
            
            print(f"\n" + "=" * 60)
            print(f"PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average rate: {len(x_coords) / total_time:.0f} points/second")
            print(f"Memory efficiency: {len(x_coords) * 24 / (1024**2):.1f} MB coordinates processed")
            print("=" * 60)
        else:
            print("No coordinates were processed successfully.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
