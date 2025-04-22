%% RRI Project Diagnostics
% This script checks the project structure and verifies that all necessary files exist

%% Get current working directory
current_dir = pwd;
fprintf('Current working directory: %s\n', current_dir);

%% Check main files
fprintf('\n--- Checking main files ---\n');
main_files = {'main.m', 'main_steps_5_to_7.m', 'main_RRI.m'};
for i = 1:length(main_files)
    file_path = fullfile(current_dir, main_files{i});
    if exist(file_path, 'file')
        fprintf('✓ %s exists\n', main_files{i});
    else
        fprintf('✗ %s does not exist\n', main_files{i});
    end
end

%% Check src directory and files
fprintf('\n--- Checking src directory and files ---\n');
src_dir = fullfile(current_dir, 'src');
if exist(src_dir, 'dir')
    fprintf('✓ src directory exists\n');
    
    % Required source files
    src_files = {'DT_setup.m', 'RRI.m', 'compute_reliability.m', 'compute_robustness.m', ...
                'evaluate_model.m', 'export_for_dl.m', 'generate_dataset.m', 'limit_states.m', ...
                'train_rapid_rri_net.m'};
    
    for i = 1:length(src_files)
        file_path = fullfile(src_dir, src_files{i});
        if exist(file_path, 'file')
            fprintf('  ✓ %s exists\n', src_files{i});
        else
            fprintf('  ✗ %s does not exist\n', src_files{i});
        end
    end
else
    fprintf('✗ src directory does not exist\n');
end

%% Check data directories
fprintf('\n--- Checking data directories ---\n');
data_dirs = {'Data', 'Data/Inputs', 'Data/Outputs', 'Models', 'Figures', 'ml_dataset'};
for i = 1:length(data_dirs)
    dir_path = fullfile(current_dir, data_dirs{i});
    if exist(dir_path, 'dir')
        fprintf('✓ %s directory exists\n', data_dirs{i});
    else
        fprintf('✗ %s directory does not exist\n', data_dirs{i});
    end
end

%% Check output data files
fprintf('\n--- Checking output data files ---\n');
output_files = {'Data/Outputs/eval_results.mat', 'Data/Outputs/mock_model.mat', ...
               'Data/Outputs/rri_dataset.mat', 'Data/Outputs/RRI_results.mat'};
for i = 1:length(output_files)
    file_path = fullfile(current_dir, output_files{i});
    if exist(file_path, 'file')
        fprintf('✓ %s exists\n', output_files{i});
    else
        fprintf('✗ %s does not exist\n', output_files{i});
    end
end

%% Add paths and verify
fprintf('\n--- Adding paths and verifying ---\n');
addpath(src_dir);
addpath(fullfile(current_dir, 'Generate Acc'));
fprintf('Added paths: src and Generate Acc\n');

% Check if functions are now accessible
test_functions = {'generate_dataset', 'export_for_dl', 'train_rapid_rri_net', 'evaluate_model'};
for i = 1:length(test_functions)
    if exist(test_functions{i}, 'file') == 2  % 2 means it's an M-file on the path
        fprintf('✓ Function %s is accessible\n', test_functions{i});
    else
        fprintf('✗ Function %s is NOT accessible\n', test_functions{i});
    end
end

%% Try loading a .mat file
fprintf('\n--- Attempting to load a .mat file ---\n');
try
    test_file = fullfile(current_dir, 'Data/Outputs/rri_dataset.mat');
    if exist(test_file, 'file')
        data = load(test_file);
        fprintf('✓ Successfully loaded %s\n', test_file);
        fprintf('  File contains variables: ');
        vars = fieldnames(data);
        for i = 1:length(vars)
            if i > 1
                fprintf(', ');
            end
            fprintf('%s', vars{i});
        end
        fprintf('\n');
    else
        fprintf('✗ Could not find %s to load\n', test_file);
    end
catch ME
    fprintf('✗ Error loading data: %s\n', ME.message);
end

%% Summary
fprintf('\n--- Diagnostics Summary ---\n');
fprintf('Project root: %s\n', current_dir);
fprintf('All checks complete. Please review any ✗ marks above.\n');