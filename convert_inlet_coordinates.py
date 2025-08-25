#!/usr/bin/env python3
"""
Script to read inlet boundary coordinates and store them as separate 1D arrays in xyz.dat format
"""

import numpy as np

def read_inlet_coordinates(input_file, output_file):
    """
    Read inlet boundary coordinates and save as separate 1D arrays for X, Y, Z
    
    Parameters:
    input_file (str): Path to inlet_boundary_coordinates.txt
    output_file (str): Path to output xyz.dat file
    """
    
    print(f"Reading inlet boundary coordinates from: {input_file}")
    
    # Read the coordinates file
    coordinates = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse X Y Z coordinates
            parts = line.split()
            if len(parts) == 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    coordinates.append([x, y, z])
                except ValueError:
                    continue
    
    # Convert to numpy array for easier manipulation
    coords_array = np.array(coordinates)
    
    print(f"Total points read: {len(coordinates)}")
    
    # Extract separate arrays for X, Y, Z
    x_coords = coords_array[:, 0]
    y_coords = coords_array[:, 1]
    z_coords = coords_array[:, 2]
    
    # Write to output file
    print(f"Writing coordinates to: {output_file}")
    
    with open(output_file, 'w') as f:
        # Write header with number of points
        f.write(f"# Inlet boundary coordinates converted from {input_file}\n")
        f.write(f"# Total points: {len(coordinates)}\n")
        f.write(f"# Format: X Y Z coordinates\n")
        f.write("\n")
        
        # Write all coordinates
        for i in range(len(coordinates)):
            f.write(f"{x_coords[i]:.6f}\t{y_coords[i]:.6f}\t{z_coords[i]:.6f}\n")
    
    # Print statistics
    print(f"\nCoordinate Statistics:")
    print(f"X: min={x_coords.min():.6f}, max={x_coords.max():.6f}")
    print(f"Y: min={y_coords.min():.6f}, max={y_coords.max():.6f}")
    print(f"Z: min={z_coords.min():.6f}, max={z_coords.max():.6f}")
    
    return x_coords, y_coords, z_coords

def save_separate_arrays(x_coords, y_coords, z_coords, prefix="inlet_coords"):
    """
    Save X, Y, Z coordinates as separate files
    
    Parameters:
    x_coords, y_coords, z_coords: numpy arrays with coordinates
    prefix (str): prefix for output filenames
    """
    
    # Save as separate text files
    np.savetxt(f"{prefix}_x.txt", x_coords, fmt='%.6f')
    np.savetxt(f"{prefix}_y.txt", y_coords, fmt='%.6f')
    np.savetxt(f"{prefix}_z.txt", z_coords, fmt='%.6f')
    
    print(f"\nSeparate coordinate arrays saved:")
    print(f"X coordinates: {prefix}_x.txt")
    print(f"Y coordinates: {prefix}_y.txt")
    print(f"Z coordinates: {prefix}_z.txt")
    
    # Also save as numpy binary files for faster loading
    np.save(f"{prefix}_x.npy", x_coords)
    np.save(f"{prefix}_y.npy", y_coords)
    np.save(f"{prefix}_z.npy", z_coords)
    
    print(f"\nNumPy binary files also saved:")
    print(f"X coordinates: {prefix}_x.npy")
    print(f"Y coordinates: {prefix}_y.npy")
    print(f"Z coordinates: {prefix}_z.npy")

if __name__ == "__main__":
    # Input and output file paths
    input_file = "inlet_boundary_coordinates.txt"
    output_file = "xyz.dat"
    
    # Read and convert coordinates
    x_coords, y_coords, z_coords = read_inlet_coordinates(input_file, output_file)
    
    # Save as separate arrays
    save_separate_arrays(x_coords, y_coords, z_coords)
    
    print(f"\nConversion completed successfully!")
    print(f"Combined coordinates saved to: {output_file}")
    print(f"Separate 1D arrays saved with prefix: inlet_coords")
