%% Main Script for Steps 5-7 of the RRI Workflow
% This script executes steps 5, 6, and 7 of the RRI workflow:
% 5. Generate Dataset for Deep Learning
% 6. Train Rapid-RRI-Net
% 7. Evaluation & Inference

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

%% Initialize
% Note: Path settings are handled by main.m
% When running this script directly, we need to add paths
if ~exist('current_dir', 'var')
    % This means we're running the script directly, not from main.m
    % Reset MATLAB path to default
    restoredefaultpath;
    rehash toolboxcache;
    
    % Get current directory
    current_dir = pwd;
    
    % Add necessary paths with full paths to avoid conflicts
    addpath(fullfile(current_dir, 'src'));
    addpath(fullfile(current_dir, 'Generate Acc'));
    addpath("/Applications/MATLAB_R2024b.app/toolbox_Stabil");
    
    % Clear workspace variables, but keep paths
    clear -except current_dir;
    clc;
    close all;
    
    fprintf('Added required paths for direct execution\n');
end

% Create required directories if they don't exist
required_dirs = {'Data', 'Data/Inputs', 'Data/Outputs', ...
                'Figures', 'Models', 'ml_dataset'};
for i = 1:length(required_dirs)
    if ~exist(required_dirs{i}, 'dir')
        mkdir(required_dirs{i});
        fprintf('Created directory: %s\n', required_dirs{i});
    end
end

%% Step 5: Generate Dataset for Deep Learning
fprintf('\n=========================================================\n');
fprintf('Step 5: Generate Dataset for Deep Learning\n');
fprintf('=========================================================\n\n');

% Configure dataset generation
dataset_options = struct();
dataset_options.num_samples = 1000;  % Default: 10000, reduced for demo
dataset_options.save_path = 'Data/Inputs/rri_dataset.mat';

% Check if dataset already exists
if exist(dataset_options.save_path, 'file')
    fprintf('Dataset already exists at %s\n', dataset_options.save_path);
    fprintf('Do you want to regenerate it? (y/n): ');
    user_input = input('', 's');
    if strcmpi(user_input, 'y')
        % Generate dataset
        dataset = generate_dataset(dataset_options.num_samples, dataset_options.save_path);
    else
        % Load existing dataset
        fprintf('Loading existing dataset...\n');
        load(dataset_options.save_path, 'dataset');
    end
else
    % Generate dataset
    dataset = generate_dataset(dataset_options.num_samples, dataset_options.save_path);
end

% Export dataset for deep learning
fprintf('\nExporting dataset for deep learning...\n');
export_options = struct();
export_options.split_ratio = [0.7, 0.15, 0.15];  % Train/val/test split
[X_train, X_val, X_test, Y_train, Y_val, Y_test, metadata] = ...
    export_for_dl(dataset_options.save_path, 'ml_dataset', export_options.split_ratio);

fprintf('\nDataset generation and export complete.\n');

%% Step 6: Train Rapid-RRI-Net
fprintf('\n=========================================================\n');
fprintf('Step 6: Train Rapid-RRI-Net\n');
fprintf('=========================================================\n\n');

% Configure model training
model_options = struct();
model_options.epochs = 50;         % Default: 100, reduced for demo
model_options.batch_size = 32;
model_options.learning_rate = 0.001;
model_options.patience = 10;
model_options.model_save_path = 'Models/rapid_rri_net.mat';

% Check if model already exists
if exist(model_options.model_save_path, 'file')
    fprintf('Model already exists at %s\n', model_options.model_save_path);
    fprintf('Do you want to retrain it? (y/n): ');
    user_input = input('', 's');
    if strcmpi(user_input, 'y')
        % Train model
        [model, history] = train_rapid_rri_net('ml_dataset', model_options);
    else
        % Load existing model
        fprintf('Loading existing model...\n');
        load(model_options.model_save_path, 'model', 'history');
    end
else
    % Train model
    [model, history] = train_rapid_rri_net('ml_dataset', model_options);
end

fprintf('\nModel training complete.\n');

%% Step 7: Evaluation & Inference
fprintf('\n=========================================================\n');
fprintf('Step 7: Evaluation & Inference\n');
fprintf('=========================================================\n\n');

% Configure evaluation
eval_options = struct();
eval_options.metrics = {'MAE', 'RMSE', 'R2'};
eval_options.noise_level = 0.05;  % Add 5% noise for robustness testing
eval_options.n_trials = 5;        % Number of trials for speed testing
eval_options.output_path = 'Data/Outputs/eval_results.mat';
eval_options.plot_confusion = true;
eval_options.plot_trajectory = true;

% Prepare test data structure
test_data = struct();
test_data.X_test = X_test;
test_data.y_test = Y_test;
test_data.history = history;  % Include training history for trajectory plotting

% Evaluate model
fprintf('Evaluating Rapid-RRI-Net model...\n');
[results] = evaluate_model(model_options.model_save_path, test_data, eval_options);

% Display evaluation results
fprintf('\nEvaluation Results:\n');
fprintf('--------------------------------------------------\n');
fprintf('MAE: %.6f\n', results.metrics.mae);
fprintf('RMSE: %.6f\n', results.metrics.rmse);
fprintf('R-squared: %.6f\n', results.metrics.r2);
fprintf('Average inference speed: %.2f samples/second\n', results.speed.mean);
if isfield(results, 'robustness')
    fprintf('Robustness (with %.1f%% noise):\n', eval_options.noise_level * 100);
    fprintf('  MAE degradation: %.6f\n', results.robustness.mae_degradation);
    fprintf('  RMSE degradation: %.6f\n', results.robustness.rmse_degradation);
end
fprintf('--------------------------------------------------\n');

% Make predictions on a few test samples to show examples
fprintf('\nSample Predictions:\n');
fprintf('Sample | Actual RRI | Predicted RRI | Error\n');
fprintf('-------------------------------------------\n');
num_samples_to_show = min(5, size(X_test, 1));
test_predictions = predict(model, X_test(1:num_samples_to_show,:));
for i = 1:num_samples_to_show
    fprintf('%6d | %9.4f | %12.4f | %5.4f\n', ...
        i, Y_test(i), test_predictions(i), abs(Y_test(i) - test_predictions(i)));
end

% Save final results to a MAT file
fprintf('\nSaving all results to RRI_results.mat...\n');
save('Data/Outputs/RRI_results.mat', 'model', 'history', 'results', 'metadata', 'dataset_options', 'model_options', 'eval_options');

fprintf('\nRRI workflow steps 5-7 completed successfully.\n');
fprintf('Results saved and visualizations generated in the Figures folder.\n');

% Show the figures directory
fprintf('\nGenerated figures:\n');
fig_files = dir('Figures/*.svg');
for i = 1:length(fig_files)
    fprintf('- %s\n', fig_files(i).name);
end