# Grid Processing Python Scripts

Three Python scripts to extract inlet boundary coordinates from Pointwise-generated grid files for use with [GenerateInflow-VFS3.1](https://github.com/hosseinsz93/GenerateInflow-VFS3.1).

## Prerequisites

Convert Plot3D file to Tecplot ASCII format:
1. Generate Plot3D file in Pointwise (e.g., `grid.x`)
2. Import `.x` file into Tecplot → Export as ASCII `.dat` file
3. Use the `.dat` file with these Python scripts

## Quick Start

```bash
python extract_inlet_boundary.py    # Find all boundary faces
# Check output files to identify your inlet face
python convert_to_xyz_dat.py        # Convert to xyz.dat for GenerateInflow
```

**Requirements**: `pip install numpy`

---

## Script Details

### 📊 `extract_inlet_boundary.py`

**Purpose**: Finds all 6 boundary faces to identify which one is your inlet.

**Usage**:
```bash
python extract_inlet_boundary.py
```

**Output**: Creates 6 files (one for each boundary face):
- `inlet_boundary_i_min.txt`, `inlet_boundary_i_max.txt`
- `inlet_boundary_j_min.txt`, `inlet_boundary_j_max.txt`
- `inlet_boundary_k_min.txt`, `inlet_boundary_k_max.txt`

---

### 🎯 `extract_inlet_simple.py`

**Purpose**: Extracts a specific inlet face once you know which one it is.

**Configuration**: Edit these variables in the script:
```python
grid_file = "grid.dat"    # Your Tecplot file
inlet_face = "k_min"           # Change to your inlet face
```

**Usage**:
```bash
python extract_inlet_simple.py
```

---

### 🔄 `convert_to_xyz_dat.py`

**Purpose**: Converts inlet coordinates to xyz.dat format for CFD preprocessing.

**Input**: `inlet_boundary_k_min.txt` (from previous scripts)

**Output**: 
- `kmin_face_simple.xyz.dat` (recommended)
- `kmin_face_structured.xyz.dat`

**Usage**:
```bash
python convert_to_xyz_dat.py
```

---

## Requirements

**Dependencies**:
```bash
pip install numpy
```

**Input File**: 
- `grid.dat` - ASCII Tecplot file (converted from Pointwise Plot3D file)

---

## File Formats

**Tecplot Input** (`.dat`):
```
TITLE     = "Plot3D DataSet"  
VARIABLES = "X" "Y" "Z"
ZONE T="grid.test.x:1"
I=101, J=101, K=501, ZONETYPE=Ordered
DATAPACKING=POINT
```

**XYZ.DAT Output**:
```
101 101 501           # Grid dimensions
0.000000  0  0        # X coordinates
0.010000  0  0
...
0  0.000000  0        # Y coordinates  
0  0.010000  0
...
0  0  0.000000        # Z coordinates
0  0  0.010000
...
```

---

**Use the generated `kmin_face_simple.xyz.dat` file with [GenerateInflow-VFS3.1](https://github.com/hosseinsz93/GenerateInflow-VFS3.1) for turbulent inflow generation.**
