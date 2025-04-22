%% Export Dataset for Deep Learning
% This script exports the generated dataset in a format suitable for deep learning
% applications (step 5 of the RRI workflow)

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

function [X_train, X_val, X_test, Y_train, Y_val, Y_test, metadata] = export_for_dl(dataset_path, export_dir, split_ratio)
    % Export the RRI dataset for deep learning in appropriate format
    %
    % Inputs:
    %   dataset_path: Path to the generated dataset (default: 'Data/Inputs/rri_dataset.mat')
    %   export_dir: Directory to export the files (default: 'ml_dataset')
    %   split_ratio: Train/val/test split ratio [train, val, test] (default: [0.7, 0.15, 0.15])
    %
    % Outputs:
    %   X_train: Training features
    %   X_val: Validation features
    %   X_test: Test features
    %   Y_train: Training targets
    %   Y_val: Validation targets
    %   Y_test: Test targets
    %   metadata: Metadata with parameter values and dataset statistics
    
    % Set defaults
    if nargin < 1
        dataset_path = 'Data/Inputs/rri_dataset.mat';
    end
    if nargin < 2
        export_dir = 'ml_dataset';
    end
    if nargin < 3
        split_ratio = [0.7, 0.15, 0.15];
    end
    
    % Validate split ratio
    if abs(sum(split_ratio) - 1) > 1e-10
        error('Split ratio must sum to 1, got: [%.2f, %.2f, %.2f]', split_ratio(1), split_ratio(2), split_ratio(3));
    end
    
    % Ensure export directory exists
    if ~isfolder(export_dir)
        mkdir(export_dir);
        fprintf('Created export directory: %s\n', export_dir);
    end
    
    % Load dataset
    fprintf('Loading dataset from %s...\n', dataset_path);
    try
        load(dataset_path, 'dataset');
    catch ME
        error('Failed to load dataset: %s', ME.message);
    end
    
    % Check dataset structure
    if ~isfield(dataset, 'features') || ~isfield(dataset, 'metadata') || ~isfield(dataset, 'targets')
        error('Dataset is missing required fields (features, metadata, targets)');
    end
    
    % Remove invalid samples (those with NaN values)
    valid_mask = ~isnan(dataset.targets.RRI);
    fprintf('Found %d valid samples out of %d total\n', sum(valid_mask), length(valid_mask));
    
    % Features: concatenate all feature types
    features = [
        dataset.features.displacements(valid_mask, :), ...
        dataset.features.rotations(valid_mask, :), ...
        dataset.features.stresses(valid_mask, :), ...
        dataset.features.frequencies(valid_mask, :)
    ];
    
    % Metadata: extract all parameter values
    metadata_values = [
        dataset.metadata.E(valid_mask), ...
        dataset.metadata.rho(valid_mask), ...
        dataset.metadata.RH(valid_mask), ...
        dataset.metadata.T(valid_mask), ...
        dataset.metadata.qw(valid_mask), ...
        dataset.metadata.deg(valid_mask)
    ];
    
    % Targets: use RRI as primary target
    targets = dataset.targets.RRI(valid_mask);
    
    % Normalize features and targets
    fprintf('Normalizing features and targets...\n');
    % For features, use z-score normalization
    features_mean = mean(features);
    features_std = std(features);
    features_norm = (features - features_mean) ./ features_std;
    
    % For metadata, use min-max normalization
    metadata_min = min(metadata_values);
    metadata_max = max(metadata_values);
    metadata_norm = (metadata_values - metadata_min) ./ (metadata_max - metadata_min);
    
    % For targets, use min-max normalization
    targets_min = min(targets);
    targets_max = max(targets);
    targets_norm = (targets - targets_min) ./ (targets_max - targets_min);
    
    % Store normalization parameters in metadata
    metadata = struct();
    metadata.features_mean = features_mean;
    metadata.features_std = features_std;
    metadata.metadata_min = metadata_min;
    metadata.metadata_max = metadata_max;
    metadata.targets_min = targets_min;
    metadata.targets_max = targets_max;
    metadata.feature_names = {
        'displacement_1', 'displacement_2', 'displacement_3', 'displacement_4', 'displacement_5', ...
        'displacement_6', 'displacement_7', 'displacement_8', 'displacement_9', 'displacement_10', ...
        'rotation_1', 'rotation_2', 'rotation_3', 'rotation_4', 'rotation_5', ...
        'rotation_6', 'rotation_7', 'rotation_8', 'rotation_9', 'rotation_10', ...
        'stress_1', 'stress_2', 'stress_3', 'stress_4', 'stress_5', ...
        'stress_6', 'stress_7', 'stress_8', 'stress_9', 'stress_10', ...
        'frequency_1', 'frequency_2', 'frequency_3', 'frequency_4', 'frequency_5', 'frequency_6', ...
        'frequency_7', 'frequency_8', 'frequency_9', 'frequency_10', 'frequency_11', 'frequency_12'
    };
    metadata.metadata_names = {'E', 'rho', 'RH', 'T', 'qw', 'deg'};
    metadata.target_names = {'RRI'};
    metadata.dataset_stats = dataset.stats;
    
    % Split the dataset into train/val/test sets
    n_samples = size(features_norm, 1);
    n_train = round(n_samples * split_ratio(1));
    n_val = round(n_samples * split_ratio(2));
    n_test = n_samples - n_train - n_val;
    
    % Create random permutation for splitting
    rng(42); % For reproducibility
    perm = randperm(n_samples);
    
    % Split indices
    train_idx = perm(1:n_train);
    val_idx = perm(n_train+1:n_train+n_val);
    test_idx = perm(n_train+n_val+1:end);
    
    % Split features
    X_train = features_norm(train_idx, :);
    X_val = features_norm(val_idx, :);
    X_test = features_norm(test_idx, :);
    
    % Split metadata
    metadata_train = metadata_norm(train_idx, :);
    metadata_val = metadata_norm(val_idx, :);
    metadata_test = metadata_norm(test_idx, :);
    
    % Split targets
    Y_train = targets_norm(train_idx);
    Y_val = targets_norm(val_idx);
    Y_test = targets_norm(test_idx);
    
    % Save the splits to CSV files for easier use in Python or other ML frameworks
    fprintf('Exporting dataset to CSV files in %s...\n', export_dir);
    
    % Features files
    writematrix(X_train, fullfile(export_dir, 'train_features.csv'));
    writematrix(X_val, fullfile(export_dir, 'val_features.csv'));
    writematrix(X_test, fullfile(export_dir, 'test_features.csv'));
    
    % Metadata files
    writematrix(metadata_train, fullfile(export_dir, 'train_metadata.csv'));
    writematrix(metadata_val, fullfile(export_dir, 'val_metadata.csv'));
    writematrix(metadata_test, fullfile(export_dir, 'test_metadata.csv'));
    
    % Target files
    writematrix(Y_train, fullfile(export_dir, 'train_targets.csv'));
    writematrix(Y_val, fullfile(export_dir, 'val_targets.csv'));
    writematrix(Y_test, fullfile(export_dir, 'test_targets.csv'));
    
    % Save metadata structure for reference
    save(fullfile(export_dir, 'metadata.mat'), 'metadata');
    
    % Create a simple README file explaining the dataset
    readme_path = fullfile(export_dir, 'README.md');
    fid = fopen(readme_path, 'w');
    if fid ~= -1
        fprintf(fid, '# RRI Dataset for Deep Learning\n\n');
        fprintf(fid, 'This dataset contains normalized features, metadata, and targets for training a deep learning model to predict the Reliability-Robustness Index (RRI) of structural systems.\n\n');
        fprintf(fid, '## Files\n\n');
        fprintf(fid, '- `train_features.csv`, `val_features.csv`, `test_features.csv`: Normalized structural response features\n');
        fprintf(fid, '- `train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`: Normalized parameter values\n');
        fprintf(fid, '- `train_targets.csv`, `val_targets.csv`, `test_targets.csv`: Normalized RRI values\n');
        fprintf(fid, '- `metadata.mat`: MATLAB file containing normalization parameters and other metadata\n\n');
        fprintf(fid, '## Dataset Statistics\n\n');
        fprintf(fid, '- Total valid samples: %d\n', n_samples);
        fprintf(fid, '- Training samples: %d (%.1f%%)\n', n_train, 100*n_train/n_samples);
        fprintf(fid, '- Validation samples: %d (%.1f%%)\n', n_val, 100*n_val/n_samples);
        fprintf(fid, '- Test samples: %d (%.1f%%)\n', n_test, 100*n_test/n_samples);
        fprintf(fid, '- RRI range: [%.4f, %.4f]\n', targets_min, targets_max);
        fprintf(fid, '- RRI mean: %.4f\n', mean(targets));
        fprintf(fid, '- RRI standard deviation: %.4f\n', std(targets));
        fclose(fid);
    end
    
    fprintf('Export complete. Dataset files are in %s\n', export_dir);
    fprintf('Training samples: %d, Validation samples: %d, Test samples: %d\n', n_train, n_val, n_test);
end