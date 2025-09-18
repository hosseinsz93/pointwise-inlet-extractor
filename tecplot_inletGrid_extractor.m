clc;
clear;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TECPLOT DATA ANALYSIS - INLET BOUNDARY EXTRACTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script reads Tecplot ASCII grid files (BLOCK datapacking),
% auto-detects the structured grid dimensions (I,J,K), and extracts the
% inlet boundary (minimum-k plane) coordinates and indices. It supports two
% coordinate-mapping conventions (Y-normal and Z-normal) and writes CSV
% outputs suitable for downstream processing (also compatible with the
% Python converters in this repository).
%
% Usage: set the `filename` variable below or call the script from MATLAB
% with the file on the path. Designed for interactive use and simple batch
% processing inside this repository.
%
% Author: Hossein Seyedzadeh
% Last updated: 2025-08-26
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start the timer
tic;

%% 2. TECPLOT FILE PARAMETERS
filename = './grid3mm.dat';  % <-- change to your actual file name

% Grid dimensions will be detected automatically from file header
Nx = [];  % Will be auto-detected
Ny = [];  % Will be auto-detected  
Nz = [];  % Will be auto-detected

% Define variable names and types
var_names = {
    'X', 'Y', 'Z'                   
};
num_vars = length(var_names);
num_nodevars = 3;         % Number of node-centered variables

orientation = 0; % orientation = 0 (y-normal), orientation = 1 (z-normal)

%% 3. LOAD TECPLOT DATA
fprintf('Loading Tecplot data from: %s\n', filename);

% Open and read the file
fid = fopen(filename, 'r');
if fid == -1
    error('❌ Failed to open file: %s', filename);
end

% Read all lines as text
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = lines{1};

% Auto-detect grid dimensions from header
fprintf('Auto-detecting grid dimensions...\n');
for i = 1:length(lines)
    line = lines{i};
    
    % Look for ZONE line with I=, J=, K= dimensions
    if contains(upper(line), 'ZONE') && contains(upper(line), 'I=')
        % Parse I=, J=, K= values
        tokens = regexp(line, '[IJK]=(\d+)', 'tokens');
        if length(tokens) >= 3
            Nx = str2double(tokens{1}{1});  % I dimension
            Ny = str2double(tokens{2}{1});  % J dimension  
            Nz = str2double(tokens{3}{1});  % K dimension
            fprintf('  Detected grid dimensions from ZONE line:\n');
            fprintf('    I=%d, J=%d, K=%d\n', Nx, Ny, Nz);
            break;
        end
    end
    
    % Alternative: Look for separate I=, J=, K= lines
    if contains(upper(line), 'I=') && contains(upper(line), 'J=') && contains(upper(line), 'K=')
        % Extract dimensions using regex
        i_match = regexp(line, 'I=(\d+)', 'tokens');
        j_match = regexp(line, 'J=(\d+)', 'tokens');
        k_match = regexp(line, 'K=(\d+)', 'tokens');
        
        if ~isempty(i_match) && ~isempty(j_match) && ~isempty(k_match)
            Nx = str2double(i_match{1}{1});
            Ny = str2double(j_match{1}{1});
            Nz = str2double(k_match{1}{1});
            fprintf('  Detected grid dimensions from header line:\n');
            fprintf('    I=%d, J=%d, K=%d\n', Nx, Ny, Nz);
            break;
        end
    end
end

% Validate that dimensions were found
if isempty(Nx) || isempty(Ny) || isempty(Nz)
    error('❌ Could not auto-detect grid dimensions. Please check file format.');
end

% Calculate derived grid parameters
N_node = Nx * Ny * Nz;    % Total number of nodes
fprintf('  Total grid points: %d\n', N_node);

% Find where numeric data starts (skip header)
data_start_idx = find(~cellfun(@isempty, regexp(lines, '^[\s\-0-9\.E\+]+$')), 1);
data_lines = lines(data_start_idx:end);
fprintf('  Numeric data starts at line %d\n', data_start_idx);

% Convert text data to numbers
data_str = strjoin(data_lines, ' ');
data_numbers = sscanf(data_str, '%f');

%% 4. PROCESS TECPLOT DATA
fprintf('Processing Tecplot data...\n');

% Separate data into variables
data = cell(1, num_vars);
offset = 0;

% Extract node-centered variables
for i = 1:num_nodevars
    data{i} = data_numbers(offset + 1 : offset + N_node);
    offset = offset + N_node;
end

% Extract cell-centered variables
for i = num_nodevars+1 : num_vars
    data{i} = data_numbers(offset + 1 : offset + N_cell);
    offset = offset + N_cell;
end

% Reshape node-centered variables into 3D arrays
X = reshape(data{1}, [Nx, Ny, Nz]);
Y = reshape(data{2}, [Nx, Ny, Nz]);
Z = reshape(data{3}, [Nx, Ny, Nz]);

fprintf('✅ Tecplot data successfully loaded.\n');
fprintf('   Grid size: %d x %d x %d (nodes)\n', Nx, Ny, Nz);
fprintf('   Cell-centered size: %d x %d x %d\n', Nx-1, Ny-1, Nz-1);

%% EXTRACT INLET BOUNDARY (k=1) AND WRITE TO CSV
fprintf('\n--- EXTRACTING INLET BOUNDARY ---\n');

% Extract inlet boundary at k=1 (minimum X location)
k_inlet = 1;  % MATLAB indexing starts from 1
inlet_data = [];

% Collect all (i,j) points at the inlet boundary
for j = 1:Ny
    for i = 1:Nx
        x_coord = X(i, j, k_inlet);
        y_coord = Y(i, j, k_inlet);
        z_coord = Z(i, j, k_inlet);
        
        % Store: [i, j, k, X, Y, Z]
        inlet_data = [inlet_data; i, j, k_inlet, x_coord, y_coord, z_coord];
    end
end

fprintf('Extracted %d inlet boundary points\n', size(inlet_data, 1));

% Display coordinate ranges for inlet boundary
fprintf('Inlet boundary coordinate ranges:\n');
fprintf('  X: %.6f to %.6f\n', min(inlet_data(:,4)), max(inlet_data(:,4)));
fprintf('  Y: %.6f to %.6f\n', min(inlet_data(:,5)), max(inlet_data(:,5)));
fprintf('  Z: %.6f to %.6f\n', min(inlet_data(:,6)), max(inlet_data(:,6)));

% % Write to CSV file
% csv_filename = 'inlet_boundary_grid.csv';
% fid = fopen(csv_filename, 'w');
% 
% % Write header
% fprintf(fid, 'i,j,k,X,Y,Z\n');
% 
% % Write data
% for row = 1:size(inlet_data, 1)
%     fprintf(fid, '%d,%d,%d,%.10e,%.10e,%.10e\n', ...
%         inlet_data(row,1), inlet_data(row,2), inlet_data(row,3), ...
%         inlet_data(row,4), inlet_data(row,5), inlet_data(row,6));
% end
% 
% fclose(fid);
% fprintf('✅ Inlet boundary data written to: %s\n', csv_filename);
% 
% % Write second CSV file with just i, j, k coordinates in separate sections
% indices_filename = 'inlet_boundary_indices.csv';
% fid2 = fopen(indices_filename, 'w');
% 
% % Write header for indices file
% fprintf(fid2, 'i,j,k\n');
% 
% % Section 1: Vary i (1 to Nx), keep j=1, k=1
% for i = 1:Nx
%     fprintf(fid2, '%d,%d,%d\n', i, 1, 1);
% end
% 
% % Section 2: Vary j (1 to Ny), keep i=1, k=1  
% for j = 1:Ny
%     fprintf(fid2, '%d,%d,%d\n', 1, j, 1);
% end
% 
% % Section 3: Vary k (1 to Nz), keep i=1, j=1
% for k = 1:Nz
%     fprintf(fid2, '%d,%d,%d\n', 1, 1, k);
% end
% 
% fclose(fid2);
% fprintf('✅ Inlet boundary indices written to: %s\n', indices_filename);

% Write third CSV file with coordinates corresponding to the indices
coords_filename = 'xyz.dat';
fid3 = fopen(coords_filename, 'w');

% Write header for coordinates file
% if orientation == 1
%     fprintf(fid3, '%d,%d,%d\n', Nz, Ny, Nx);
% else
%         fprintf(fid3, '%d,%d,%d\n', Nx, Ny, Nz);
% end
fprintf(fid3, '%d,%d,%d\n', Nx, Ny, Nz);

% Section 1: Coordinates for varying i (1 to Nx), keep j=1, k=1
if orientation == 1
for i = 1:Nx
    x_coord = Y(i, 1, 1);
    y_coord = Z(i, 1, 1);
    z_coord = X(i, 1, 1);
    
    fprintf(fid3, '%.10e %.10e %.10e\n', x_coord, y_coord, z_coord);
end

% Section 2: Coordinates for varying j (1 to Ny), keep i=1, k=1  
for j = 1:Ny
    x_coord = Y(1, j, 1);
    y_coord = Z(1, j, 1);
    z_coord = X(1, j, 1);
    fprintf(fid3, '%.10e %.10e %.10e\n', x_coord, y_coord, z_coord);
end

% Section 3: Coordinates for varying k (1 to Nz), keep i=1, j=1
for k = 1:Nz
    x_coord = Y(1, 1, 1);
    y_coord = Z(1, 1, 1);
    z_coord = X(1, 1, 1);
    fprintf(fid3, '%.10e %.10e %.10e\n', x_coord, y_coord, z_coord);
end

else
for i = 1:Nx
    x_coord = X(i, 1, 1);
    y_coord = Y(i, 1, 1);
    z_coord = Z(i, 1, 1);
    fprintf(fid3, '%.10e %.10e %.10e\n', x_coord, y_coord, z_coord);
end

% Section 2: Coordinates for varying j (1 to Ny), keep i=1, k=1  
for j = 1:Ny
    x_coord = X(1, j, 1);
    y_coord = Y(1, j, 1);
    z_coord = Z(1, j, 1);
    fprintf(fid3, '%.10e %.10e %.10e\n', x_coord, y_coord, z_coord);
end

% Section 3: Coordinates for varying k (1 to Nz), keep i=1, j=1
for k = 1:Nz
    x_coord = X(1, 1, 1);
    y_coord = Y(1, 1, 1);
    z_coord = Z(1, 1, 1);
    fprintf(fid3, '%.10e %.10e %.10e\n', x_coord, y_coord, z_coord);
end
end

fclose(fid3);
fprintf('✅ Corresponding coordinates written to: %s\n', coords_filename);

%% Calculate and display execution time
elapsed_time = toc;
fprintf('\n=== EXECUTION SUMMARY ===\n');
fprintf('Script execution completed successfully.\n');
fprintf('Total execution time: %.3f seconds (%.2f minutes)\n', elapsed_time, elapsed_time/60);
if elapsed_time > 60
    minutes = floor(elapsed_time/60);
    seconds = mod(elapsed_time, 60);
    fprintf('Formatted time: %d minutes %.1f seconds\n', minutes, seconds);
end
fprintf('========================\n');

%% End of script



