#!/usr/bin/env python3
"""
Parallel version of inlet boundary extraction script.
Uses multiprocessing to handle large grid files efficiently.
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import os
from functools import partial
import psutil

def parse_coordinate_chunk(chunk_data):
    """
    Parse a chunk of coordinate lines in parallel
    
    Args:
        chunk_data (tuple): (lines, start_index)
    
    Returns:
        tuple: (coordinates_array, start_index)
    """
    lines, start_idx = chunk_data
    coords = []
    
    for line in lines:
        line = line.strip()
        if line:
            values = line.split()
            if len(values) >= 3:
                try:
                    coords.append([float(values[0]), float(values[1]), float(values[2])])
                except ValueError:
                    continue
    
    return np.array(coords), start_idx

def read_grid_file_parallel(grid_file, num_workers=None):
    """
    Read Tecplot grid file in parallel
    
    Args:
        grid_file (str): Path to the grid file
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (coordinates_array, dimensions)
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)
    
    print(f"Reading grid file in parallel with {num_workers} workers: {grid_file}")
    
    # Read file and find dimensions
    with open(grid_file, 'r') as f:
        lines = f.readlines()
    
    # Extract grid dimensions from header
    i_dim = j_dim = k_dim = None
    data_start = 0
    
    for i, line in enumerate(lines):
        if 'I=' in line and 'J=' in line and 'K=' in line:
            parts = line.split()
            for part in parts:
                if part.startswith('I='):
                    i_dim = int(part.split('=')[1].rstrip(','))
                elif part.startswith('J='):
                    j_dim = int(part.split('=')[1].rstrip(','))
                elif part.startswith('K='):
                    k_dim = int(part.split('=')[1].rstrip(','))
        if 'DT=' in line:
            data_start = i + 1
            break
    
    print(f"Grid dimensions: I={i_dim}, J={j_dim}, K={k_dim}")
    
    # Split data lines into chunks for parallel processing
    data_lines = lines[data_start:]
    chunk_size = max(1000, len(data_lines) // (num_workers * 4))  # Ensure enough chunks
    
    chunks = []
    for i in range(0, len(data_lines), chunk_size):
        chunk = data_lines[i:i + chunk_size]
        chunks.append((chunk, i))
    
    print(f"Processing {len(data_lines)} lines in {len(chunks)} chunks")
    
    # Process chunks in parallel
    start_time = time.time()
    all_coords = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(parse_coordinate_chunk, chunk): idx 
                          for idx, chunk in enumerate(chunks)}
        
        # Collect results in order
        chunk_results = [None] * len(chunks)
        
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                coords, start_idx = future.result()
                chunk_results[chunk_idx] = coords
                print(f"Processed chunk {chunk_idx + 1}/{len(chunks)} ({len(coords)} points)")
            except Exception as exc:
                print(f"Chunk {chunk_idx} generated an exception: {exc}")
                chunk_results[chunk_idx] = np.array([])
    
    # Combine results
    valid_coords = [coords for coords in chunk_results if len(coords) > 0]
    if valid_coords:
        all_coords = np.vstack(valid_coords)
    else:
        all_coords = np.array([])
    
    parse_time = time.time() - start_time
    print(f"Parallel parsing completed in {parse_time:.2f} seconds")
    print(f"Total coordinates read: {len(all_coords)}")
    
    return all_coords, (i_dim, j_dim, k_dim)

def extract_inlet_boundary_parallel(grid_file, inlet_face='k_min', num_workers=None):
    """
    Extract inlet boundary coordinates from Tecplot grid file using parallel processing.
    
    Args:
        grid_file (str): Path to the Tecplot data file
        inlet_face (str): Which face is the inlet ('i_min', 'i_max', 'j_min', 'j_max', 'k_min', 'k_max')
        num_workers (int): Number of worker processes
    
    Returns:
        np.array: Array of inlet boundary coordinates [N, 3] where N is number of points
    """
    
    start_time = time.time()
    
    # Read coordinates in parallel
    coords, (i_dim, j_dim, k_dim) = read_grid_file_parallel(grid_file, num_workers)
    
    if len(coords) == 0:
        raise ValueError("No coordinates found in the grid file")
    
    # Reshape coordinates to 3D grid
    print("Reshaping coordinates to 3D grid...")
    reshape_start = time.time()
    
    expected_points = i_dim * j_dim * k_dim
    if len(coords) != expected_points:
        print(f"Warning: Expected {expected_points} points but got {len(coords)}")
        # Adjust dimensions if necessary
        actual_points = len(coords)
        if actual_points < expected_points:
            print(f"Trimming dimensions to match available data")
    
    try:
        coords_3d = coords.reshape((i_dim, j_dim, k_dim, 3))
    except ValueError as e:
        print(f"Reshape error: {e}")
        print(f"Trying to reshape {len(coords)} points to ({i_dim}, {j_dim}, {k_dim}, 3)")
        raise
    
    reshape_time = time.time() - reshape_start
    print(f"Reshape completed in {reshape_time:.2f} seconds")
    
    # Extract inlet boundary based on specified face
    extract_start = time.time()
    
    if inlet_face == 'i_min':
        inlet_coords = coords_3d[0, :, :, :].reshape(-1, 3)
        print(f"Extracted inlet boundary from I=1 face: {len(inlet_coords)} points")
    elif inlet_face == 'i_max':
        inlet_coords = coords_3d[-1, :, :, :].reshape(-1, 3)
        print(f"Extracted inlet boundary from I={i_dim} face: {len(inlet_coords)} points")
    elif inlet_face == 'j_min':
        inlet_coords = coords_3d[:, 0, :, :].reshape(-1, 3)
        print(f"Extracted inlet boundary from J=1 face: {len(inlet_coords)} points")
    elif inlet_face == 'j_max':
        inlet_coords = coords_3d[:, -1, :, :].reshape(-1, 3)
        print(f"Extracted inlet boundary from J={j_dim} face: {len(inlet_coords)} points")
    elif inlet_face == 'k_min':
        inlet_coords = coords_3d[:, :, 0, :].reshape(-1, 3)
        print(f"Extracted inlet boundary from K=1 face: {len(inlet_coords)} points")
    elif inlet_face == 'k_max':
        inlet_coords = coords_3d[:, :, -1, :].reshape(-1, 3)
        print(f"Extracted inlet boundary from K={k_dim} face: {len(inlet_coords)} points")
    else:
        raise ValueError(f"Unknown inlet face: {inlet_face}")
    
    extract_time = time.time() - extract_start
    total_time = time.time() - start_time
    
    print(f"Boundary extraction completed in {extract_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return inlet_coords

def save_inlet_coordinates_parallel(coords, output_file="inlet_boundary_coordinates.txt", 
                                  chunk_size=10000):
    """
    Save inlet coordinates to a text file using parallel writing for large datasets.
    
    Args:
        coords (np.array): Coordinates array
        output_file (str): Output filename
        chunk_size (int): Number of lines to write per chunk
    """
    
    print(f"Saving {len(coords)} coordinates to: {output_file}")
    start_time = time.time()
    
    # For smaller datasets, use simple sequential writing
    if len(coords) < 50000:
        with open(output_file, 'w') as f:
            f.write(f"# Inlet boundary coordinates\n")
            f.write(f"# Format: X Y Z\n")
            f.write(f"# Total points: {len(coords)}\n")
            
            for coord in coords:
                f.write(f"{coord[0]:16.10e} {coord[1]:16.10e} {coord[2]:16.10e}\n")
    else:
        # For larger datasets, prepare chunks and write header
        with open(output_file, 'w') as f:
            f.write(f"# Inlet boundary coordinates\n")
            f.write(f"# Format: X Y Z\n")
            f.write(f"# Total points: {len(coords)}\n")
        
        # Prepare coordinate strings in parallel
        def format_coordinate_chunk(chunk_data):
            chunk_coords, chunk_idx = chunk_data
            formatted_lines = []
            for coord in chunk_coords:
                formatted_lines.append(f"{coord[0]:16.10e} {coord[1]:16.10e} {coord[2]:16.10e}\n")
            return formatted_lines, chunk_idx
        
        # Split coordinates into chunks
        chunks = []
        for i in range(0, len(coords), chunk_size):
            chunk = coords[i:i + chunk_size]
            chunks.append((chunk, i // chunk_size))
        
        print(f"Writing in {len(chunks)} parallel chunks...")
        
        # Format chunks in parallel
        with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
            future_to_chunk = {executor.submit(format_coordinate_chunk, chunk): idx 
                              for idx, chunk in enumerate(chunks)}
            
            # Collect formatted strings in order
            formatted_chunks = [None] * len(chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    formatted_lines, original_idx = future.result()
                    formatted_chunks[original_idx] = formatted_lines
                except Exception as exc:
                    print(f"Formatting chunk {chunk_idx} failed: {exc}")
        
        # Write all formatted chunks to file
        with open(output_file, 'a') as f:
            for chunk_lines in formatted_chunks:
                if chunk_lines:
                    f.writelines(chunk_lines)
    
    save_time = time.time() - start_time
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"File saved in {save_time:.2f} seconds ({file_size:.1f} MB)")

def run_performance_comparison(grid_file, inlet_face):
    """
    Compare performance between original and parallel versions
    """
    print("=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Import original function
    from extract_inlet_simple import extract_inlet_boundary
    
    print(f"Grid file: {grid_file}")
    print(f"Inlet face: {inlet_face}")
    print(f"Available CPU cores: {mp.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # Test original version
    print("\n1. ORIGINAL VERSION:")
    try:
        start_time = time.time()
        coords_original = extract_inlet_boundary(grid_file, inlet_face)
        original_time = time.time() - start_time
        print(f"Original processing time: {original_time:.2f} seconds")
        original_points = len(coords_original)
    except Exception as e:
        print(f"Original version failed: {e}")
        original_time = float('inf')
        original_points = 0
        coords_original = None
    
    # Test parallel version
    print("\n2. PARALLEL VERSION:")
    try:
        start_time = time.time()
        coords_parallel = extract_inlet_boundary_parallel(grid_file, inlet_face)
        parallel_time = time.time() - start_time
        print(f"Parallel processing time: {parallel_time:.2f} seconds")
        parallel_points = len(coords_parallel)
    except Exception as e:
        print(f"Parallel version failed: {e}")
        parallel_time = float('inf')
        parallel_points = 0
        coords_parallel = None
    
    # Compare results
    print(f"\n3. COMPARISON:")
    if original_time != float('inf') and parallel_time != float('inf'):
        speedup = original_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
        print(f"Performance improvement: {(speedup - 1) * 100:.1f}%")
        
        # Verify results match
        if coords_original is not None and coords_parallel is not None:
            coords_match = np.allclose(coords_original, coords_parallel, rtol=1e-10)
            print(f"Results match: {coords_match}")
            print(f"Original points: {original_points}")
            print(f"Parallel points: {parallel_points}")
    else:
        print("Could not compare due to errors in one or both versions")

def auto_detect_grid_file():
    """Auto-detect the grid file to use"""
    grid_files = [f for f in os.listdir('.') if f.endswith('.dat') and 'grid' in f.lower()]
    if grid_files:
        return grid_files[0]  # Use first grid file found
    
    # Fallback to any .dat file
    dat_files = [f for f in os.listdir('.') if f.endswith('.dat')]
    if dat_files:
        return dat_files[0]
    
    return None

def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    cpu_count = mp.cpu_count()
    
    # For very large files, limit workers to prevent memory issues
    try:
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 8:
            return min(4, cpu_count)  # Limit to 4 workers if low memory
        elif available_memory_gb < 16:
            return min(6, cpu_count)  # Limit to 6 workers if medium memory
        else:
            return min(8, cpu_count)  # Max 8 workers for high memory
    except:
        return min(4, cpu_count)  # Safe fallback

def save_inlet_coordinates_robust(coords, output_file="inlet_boundary_coordinates.txt"):
    """
    Robust save function that ensures file is written correctly.
    """
    print(f"Saving {len(coords)} coordinates to: {output_file}")
    start_time = time.time()
    
    try:
        # Get absolute path to ensure we know where it's being saved
        abs_path = os.path.abspath(output_file)
        print(f"Full path: {abs_path}")
        
        # Create backup filename if original exists
        if os.path.exists(output_file):
            backup_file = f"{output_file}.backup"
            print(f"Backing up existing file to: {backup_file}")
            os.rename(output_file, backup_file)
        
        # Write file with explicit encoding and flushing
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# Inlet boundary coordinates\n")
            f.write(f"# Format: X Y Z\n") 
            f.write(f"# Total points: {len(coords)}\n")
            f.write(f"# Generated by parallel extractor on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.flush()  # Ensure header is written
            
            # Write coordinates with progress tracking
            for i, coord in enumerate(coords):
                f.write(f"{coord[0]:16.10e} {coord[1]:16.10e} {coord[2]:16.10e}\n")
                
                # Progress indicator and periodic flushing for large files
                if (i + 1) % 5000 == 0:
                    f.flush()  # Flush to disk periodically
                    progress = ((i + 1) / len(coords)) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1:,} / {len(coords):,} points)")
            
            f.flush()  # Final flush
        
        # Verify file was created successfully
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            save_time = time.time() - start_time
            
            print(f"✅ File successfully created!")
            print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            print(f"   Save time: {save_time:.2f} seconds")
            print(f"   Location: {abs_path}")
            
            # Verify content by reading first and last lines
            with open(output_file, 'r') as f:
                lines = f.readlines()
                print(f"   Lines written: {len(lines)}")
                print(f"   Header: {lines[0].strip()}")
                if len(lines) > 4:
                    print(f"   First data: {lines[4].strip()}")
                    print(f"   Last data: {lines[-1].strip()}")
            
            return True
        else:
            print("❌ File was not created!")
            return False
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution - AGENT MODE (No user input required)
if __name__ == "__main__":
    print("🤖 AGENT MODE: PARALLEL INLET BOUNDARY EXTRACTOR")
    print("=" * 70)
    print("Running in autonomous mode with smart defaults...")
    
    start_time = time.time()
    
    # Auto-detect grid file
    grid_file = auto_detect_grid_file()
    if not grid_file:
        print("❌ No grid files found in current directory!")
        print("Available files:")
        for f in os.listdir('.'):
            print(f"  {f}")
        exit(1)
    
    # Configuration with smart defaults
    inlet_face = "k_min"  # Most common inlet face
    num_workers = get_optimal_workers()
    output_file = "inlet_boundary_coordinates.txt"
    
    # System information
    try:
        file_size_gb = os.path.getsize(grid_file) / (1024**3)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
    except:
        file_size_gb = 0
        memory_gb = 0
        available_memory_gb = 0
    
    print(f"📁 Grid file: {grid_file} ({file_size_gb:.2f} GB)")
    print(f"🎯 Inlet face: {inlet_face}")
    print(f"⚡ Workers: {num_workers} (of {mp.cpu_count()} available cores)")
    print(f"💾 System RAM: {available_memory_gb:.1f} GB available / {memory_gb:.1f} GB total")
    print(f"📄 Output file: {output_file}")
    
    try:
        print(f"\n🚀 Starting parallel extraction...")
        extraction_start = time.time()
        
        # Extract inlet boundary coordinates
        inlet_coords = extract_inlet_boundary_parallel(grid_file, inlet_face, num_workers)
        
        extraction_time = time.time() - extraction_start
        print(f"⏱️  Extraction completed in {extraction_time:.2f} seconds")
        
        # Save to file using robust method
        print(f"\n💾 Saving coordinates...")
        save_success = save_inlet_coordinates_robust(inlet_coords, output_file)
        
        if save_success:
            total_time = time.time() - start_time
            
            # Display comprehensive summary
            print(f"\n" + "🎉 " + "=" * 66)
            print(f"🎉 EXTRACTION COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"📊 Statistics:")
            print(f"   Total points extracted: {len(inlet_coords):,}")
            print(f"   Processing rate: {len(inlet_coords) / extraction_time:.0f} points/second")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Memory usage: {len(inlet_coords) * 24 / (1024**2):.1f} MB")
            
            print(f"\n📐 Coordinate ranges:")
            print(f"   X: {inlet_coords[:, 0].min():.6f} to {inlet_coords[:, 0].max():.6f}")
            print(f"   Y: {inlet_coords[:, 1].min():.6f} to {inlet_coords[:, 1].max():.6f}")
            print(f"   Z: {inlet_coords[:, 2].min():.6f} to {inlet_coords[:, 2].max():.6f}")
            
            print(f"\n📝 Output:")
            print(f"   File: {output_file}")
            print(f"   Format: Scientific notation (X Y Z per line)")
            print(f"   Ready for use in CFD applications!")
            
            # Show sample points
            print(f"\n🔍 Sample inlet points:")
            for i in range(min(5, len(inlet_coords))):
                x, y, z = inlet_coords[i]
                print(f"   Point {i+1}: X={x:12.6f}, Y={y:12.6f}, Z={z:12.6f}")
            
            print("=" * 70)
            print(f"🤖 Agent mode execution completed successfully!")
            
        else:
            print("❌ Failed to save coordinates to file!")
            exit(1)
                
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        print("\n🔧 Troubleshooting information:")
        print(f"   Grid file: {grid_file}")
        print(f"   File exists: {os.path.exists(grid_file)}")
        print(f"   File size: {file_size_gb:.2f} GB")
        print(f"   Available memory: {available_memory_gb:.1f} GB")
        print(f"   Workers: {num_workers}")
        
        import traceback
        print(f"\n📋 Full error details:")
        traceback.print_exc()
        exit(1)
