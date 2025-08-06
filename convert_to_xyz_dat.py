#!/usr/bin/env python3
"""
Convert K_min face coordinates to xyz.dat format.
The xyz.dat format stores coordinates as separate 1D arrays for X, Y, and Z.
"""

import numpy as np

def read_inlet_boundary(filename):
    """Read inlet boundary coordinates from the extracted file."""
    coords = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                values = line.split()
                if len(values) >= 3:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
                    coords.append([x, y, z])
    
    return np.array(coords)

def determine_grid_dimensions(coords):
    """
    Determine the structured grid dimensions from the coordinates.
    For a K_min face from a structured grid (I x J x K), we have I*J points.
    We need to figure out I and J dimensions.
    """
    total_points = len(coords)
    print(f"Total points: {total_points}")
    
    # Extract unique X and Y values to determine dimensions
    unique_x = np.unique(np.round(coords[:, 0], 6))  # Round to avoid floating point issues
    unique_y = np.unique(np.round(coords[:, 1], 6))
    
    ni = len(unique_x)
    nj = len(unique_y)
    
    print(f"Detected dimensions: NI={ni}, NJ={nj}")
    print(f"Expected total points: {ni * nj}")
    
    if ni * nj != total_points:
        print("Warning: Grid dimensions don't match total points!")
        print("Trying to find best fit dimensions...")
        
        # Try to find factors of total_points
        for i in range(1, int(np.sqrt(total_points)) + 1):
            if total_points % i == 0:
                j = total_points // i
                print(f"Possible dimensions: {i} x {j}")
        
        # Use the detected unique values as best guess
        ni = len(unique_x)
        nj = len(unique_y)
    
    return ni, nj, unique_x, unique_y

def write_xyz_dat(coords, ni, nj, output_filename="xyz.dat"):
    """
    Write coordinates in xyz.dat format.
    Format: 
    - First line: NI NJ NK (where NK=1 for a 2D face)
    - Then X coordinates array
    - Then Y coordinates array  
    - Then Z coordinates array
    """
    
    # For a face, NK = 1
    nk = 1
    
    # Reshape coordinates to structured grid format
    try:
        coords_2d = coords.reshape(ni, nj, 3)
    except ValueError:
        print(f"Cannot reshape {len(coords)} points to {ni}x{nj} grid")
        print("Using sequential ordering...")
        # Pad with zeros if needed
        needed_points = ni * nj
        if len(coords) < needed_points:
            padding = np.zeros((needed_points - len(coords), 3))
            coords = np.vstack([coords, padding])
        coords_2d = coords[:needed_points].reshape(ni, nj, 3)
    
    with open(output_filename, 'w') as f:
        # Write dimensions
        f.write(f"{ni} {nj} {nk}\n")
        
        # Write X coordinates
        for j in range(nj):
            for i in range(ni):
                f.write(f"{coords_2d[i, j, 0]:.6f}\t")
            f.write("\n")
        
        # Write Y coordinates  
        for j in range(nj):
            for i in range(ni):
                f.write(f"{coords_2d[i, j, 1]:.6f}\t")
            f.write("\n")
        
        # Write Z coordinates
        for j in range(nj):
            for i in range(ni):
                f.write(f"{coords_2d[i, j, 2]:.6f}\t")
            f.write("\n")
    
    print(f"K_min face coordinates written to: {output_filename}")
    print(f"Grid dimensions: {ni} x {nj} x {nk}")
    
    return coords_2d

def write_xyz_dat_simple(coords, output_filename="inlet_boundary_coordinates.txt"):
    """
    Write coordinates in a simpler xyz.dat format - just coordinate arrays.
    This format is more similar to your original xyz.dat file.
    """
    
    # Extract X, Y, Z arrays
    x_coords = coords[:, 0]
    y_coords = coords[:, 1] 
    z_coords = coords[:, 2]
    
    # Get unique values and sort them
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    unique_z = np.unique(z_coords)
    
    ni = len(unique_x)
    nj = len(unique_y)
    nk = len(unique_z)
    
    with open(output_filename, 'w') as f:
        # Write dimensions
        f.write(f"{ni} {nj} {nk}\n")
        
        # Write X coordinate array
        for x in unique_x:
            f.write(f"{x:.6f}\t0\t0\n")
        
        # Write Y coordinate array
        for y in unique_y:
            f.write(f"0\t{y:.6f}\t0\n")
        
        # Write Z coordinate array  
        for z in unique_z:
            f.write(f"0\t0\t{z:.6f}\n")
    
    print(f"K_min face coordinates (simple format) written to: {output_filename}")
    print(f"Unique coordinates: X={ni}, Y={nj}, Z={nk}")

def main():
    # Input file
    inlet_file = "inlet_boundary_coordinates.txt"
    
    try:
        # Read the inlet boundary coordinates
        print("Reading inlet boundary coordinates...")
        coords = read_inlet_boundary(inlet_file)
        
        print(f"Loaded {len(coords)} coordinate points")
        print(f"Coordinate ranges:")
        print(f"  X: {coords[:, 0].min():.6f} to {coords[:, 0].max():.6f}")
        print(f"  Y: {coords[:, 1].min():.6f} to {coords[:, 1].max():.6f}")
        print(f"  Z: {coords[:, 2].min():.6f} to {coords[:, 2].max():.6f}")
        
        # Determine grid dimensions
        ni, nj, unique_x, unique_y = determine_grid_dimensions(coords)
        
        # Write 
        print("\nWriting simple xyz.dat format...")
        write_xyz_dat_simple(coords, "xyz.dat")
        
        print("\nConversion completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{inlet_file}'")
        print("Make sure you have run the inlet boundary extraction script first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
