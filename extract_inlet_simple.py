#!/usr/bin/env python3
"""
Simplified script to extract inlet boundary coordinates.
Modify the 'inlet_face' variable to specify which boundary face is your inlet.
"""

import numpy as np

def extract_inlet_boundary(grid_file, inlet_face='k_min'):
    """
    Extract inlet boundary coordinates from Tecplot grid file.
    
    Args:
        grid_file (str): Path to the Tecplot data file
        inlet_face (str): Which face is the inlet ('i_min', 'i_max', 'j_min', 'j_max', 'k_min', 'k_max')
    
    Returns:
        np.array: Array of inlet boundary coordinates [N, 3] where N is number of points
    """
    
    print(f"Reading grid file: {grid_file}")
    
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
    
    # Read all coordinates
    coords = []
    for line_idx in range(data_start, len(lines)):
        line = lines[line_idx].strip()
        if line:
            values = line.split()
            if len(values) >= 3:
                try:
                    coords.append([float(values[0]), float(values[1]), float(values[2])])
                except ValueError:
                    continue
    
    coords = np.array(coords)
    coords_3d = coords.reshape((i_dim, j_dim, k_dim, 3))
    
    # Extract inlet boundary based on specified face
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
    
    return inlet_coords

def save_inlet_coordinates(coords, output_file="inlet_boundary_coordinates.txt"):
    """Save inlet coordinates to a text file."""
    with open(output_file, 'w') as f:
        f.write(f"# Inlet boundary coordinates\n")
        f.write(f"# Format: X Y Z\n")
        f.write(f"# Total points: {len(coords)}\n")
        
        for i, coord in enumerate(coords):
            f.write(f"{coord[0]:16.10e} {coord[1]:16.10e} {coord[2]:16.10e}\n")
    
    print(f"Inlet coordinates saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    # Configuration - MODIFY THESE VALUES
    grid_file = "grid-flood.dat"  # Your Tecplot grid file
    inlet_face = "k_min"         # Change this to the correct inlet face
                                 # Options: 'i_min', 'i_max', 'j_min', 'j_max', 'k_min', 'k_max'
    
    try:
        # Extract inlet boundary coordinates
        inlet_coords = extract_inlet_boundary(grid_file, inlet_face)
        
        # Save to file
        save_inlet_coordinates(inlet_coords)
        
        # Display summary
        print(f"\nInlet boundary summary:")
        print(f"Number of points: {len(inlet_coords)}")
        print(f"X range: {inlet_coords[:, 0].min():.6f} to {inlet_coords[:, 0].max():.6f}")
        print(f"Y range: {inlet_coords[:, 1].min():.6f} to {inlet_coords[:, 1].max():.6f}")
        print(f"Z range: {inlet_coords[:, 2].min():.6f} to {inlet_coords[:, 2].max():.6f}")
        
        # Show first few points
        print(f"\nFirst 5 inlet points:")
        for i in range(min(5, len(inlet_coords))):
            x, y, z = inlet_coords[i]
            print(f"  Point {i+1}: X={x:12.6f}, Y={y:12.6f}, Z={z:12.6f}")
            
    except Exception as e:
        print(f"Error: {e}")
