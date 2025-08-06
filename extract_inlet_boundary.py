#!/usr/bin/env python3
"""
Script to extract inlet boundary coordinates from Tecplot grid data file.
The script can extract coordinates from any boundary face of a structured grid.
"""

import numpy as np
import sys

def read_tecplot_grid(filename):
    """
    Read Tecplot grid file and extract coordinates.
    
    Args:
        filename (str): Path to the Tecplot data file
        
    Returns:
        tuple: (coordinates array, grid dimensions)
    """
    print(f"Reading grid file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the header information
    i_dim = j_dim = k_dim = None
    data_start_line = 0
    
    for i, line in enumerate(lines):
        if 'I=' in line and 'J=' in line and 'K=' in line:
            # Extract grid dimensions
            parts = line.split()
            for part in parts:
                if part.startswith('I='):
                    i_dim = int(part.split('=')[1].rstrip(','))
                elif part.startswith('J='):
                    j_dim = int(part.split('=')[1].rstrip(','))
                elif part.startswith('K='):
                    k_dim = int(part.split('=')[1].rstrip(','))
        
        # Find where data starts (after the header)
        if 'DT=' in line:
            data_start_line = i + 1
            break
    
    if not all([i_dim, j_dim, k_dim]):
        raise ValueError("Could not find grid dimensions in header")
    
    print(f"Grid dimensions: I={i_dim}, J={j_dim}, K={k_dim}")
    
    # Read coordinate data
    coords = []
    total_points = i_dim * j_dim * k_dim
    
    print(f"Reading {total_points} grid points...")
    
    for line_idx in range(data_start_line, len(lines)):
        line = lines[line_idx].strip()
        if line:
            values = line.split()
            if len(values) >= 3:
                try:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    
    coords = np.array(coords)
    
    if len(coords) != total_points:
        print(f"Warning: Expected {total_points} points, got {len(coords)}")
    
    # Reshape coordinates to structured grid format [i, j, k, xyz]
    coords_3d = coords.reshape((i_dim, j_dim, k_dim, 3))
    
    return coords_3d, (i_dim, j_dim, k_dim)

def extract_boundary_face(coords_3d, face_type, dims):
    """
    Extract coordinates from a specific boundary face.
    
    Args:
        coords_3d (np.array): 4D array of coordinates [i, j, k, xyz]
        face_type (str): Type of boundary face ('i_min', 'i_max', 'j_min', 'j_max', 'k_min', 'k_max')
        dims (tuple): Grid dimensions (i_dim, j_dim, k_dim)
        
    Returns:
        np.array: 2D array of boundary coordinates
    """
    i_dim, j_dim, k_dim = dims
    
    if face_type == 'i_min':
        # I = 0 face (first I index)
        boundary_coords = coords_3d[0, :, :, :].reshape(-1, 3)
        print(f"Extracted {len(boundary_coords)} points from I_min face (I=1)")
        
    elif face_type == 'i_max':
        # I = i_dim-1 face (last I index)
        boundary_coords = coords_3d[-1, :, :, :].reshape(-1, 3)
        print(f"Extracted {len(boundary_coords)} points from I_max face (I={i_dim})")
        
    elif face_type == 'j_min':
        # J = 0 face (first J index)
        boundary_coords = coords_3d[:, 0, :, :].reshape(-1, 3)
        print(f"Extracted {len(boundary_coords)} points from J_min face (J=1)")
        
    elif face_type == 'j_max':
        # J = j_dim-1 face (last J index)
        boundary_coords = coords_3d[:, -1, :, :].reshape(-1, 3)
        print(f"Extracted {len(boundary_coords)} points from J_max face (J={j_dim})")
        
    elif face_type == 'k_min':
        # K = 0 face (first K index)
        boundary_coords = coords_3d[:, :, 0, :].reshape(-1, 3)
        print(f"Extracted {len(boundary_coords)} points from K_min face (K=1)")
        
    elif face_type == 'k_max':
        # K = k_dim-1 face (last K index)
        boundary_coords = coords_3d[:, :, -1, :].reshape(-1, 3)
        print(f"Extracted {len(boundary_coords)} points from K_max face (K={k_dim})")
        
    else:
        raise ValueError(f"Unknown face type: {face_type}")
    
    return boundary_coords

def save_boundary_coords(coords, filename, face_type):
    """
    Save boundary coordinates to a file.
    
    Args:
        coords (np.array): Boundary coordinates
        filename (str): Output filename
        face_type (str): Type of boundary face
    """
    with open(filename, 'w') as f:
        f.write(f"# Boundary coordinates from {face_type} face\n")
        f.write(f"# Format: X Y Z\n")
        f.write(f"# Total points: {len(coords)}\n")
        
        for coord in coords:
            f.write(f"{coord[0]:16.10e} {coord[1]:16.10e} {coord[2]:16.10e}\n")
    
    print(f"Boundary coordinates saved to: {filename}")

def main():
    # Configuration
    grid_file = "grid.test.dat"  # Input Tecplot file
    
    # Available boundary faces - choose the one that represents your inlet
    available_faces = ['i_min', 'i_max', 'j_min', 'j_max', 'k_min', 'k_max']
    
    print("Available boundary faces:")
    for i, face in enumerate(available_faces):
        print(f"  {i+1}. {face}")
    
    try:
        # Read the grid
        coords_3d, dims = read_tecplot_grid(grid_file)
        
        # Extract all boundary faces and let user choose
        print("\nExtracting all boundary faces...")
        
        for face_type in available_faces:
            try:
                boundary_coords = extract_boundary_face(coords_3d, face_type, dims)
                output_filename = f"inlet_boundary_{face_type}.txt"
                save_boundary_coords(boundary_coords, output_filename, face_type)
                
                # Show sample coordinates
                print(f"Sample coordinates from {face_type}:")
                for i in range(min(5, len(boundary_coords))):
                    x, y, z = boundary_coords[i]
                    print(f"  Point {i+1}: X={x:12.6f}, Y={y:12.6f}, Z={z:12.6f}")
                print()
                
            except Exception as e:
                print(f"Error extracting {face_type}: {e}")
        
        print("All boundary faces extracted successfully!")
        print("Check the generated files and identify which one represents your inlet boundary.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
