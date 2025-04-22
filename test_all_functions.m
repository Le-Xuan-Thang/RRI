%% Test All RRI Functions
% This script tests all the functions in the RRI workflow to ensure they work correctly.

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

%% Initialize
try
    % Reset MATLAB path and add required paths
    restoredefaultpath;
    rehash toolboxcache;
    
    % Get current directory
    current_dir = pwd;
    
    % Add necessary paths
    addpath(fullfile(current_dir, 'src'));
    addpath(fullfile(current_dir, 'Generate Acc'));
    addpath("/Applications/MATLAB_R2024b.app/toolbox_Stabil");
    fprintf('Reset MATLAB path and added required folders\n');
    
    % Clean workspace and command window
    clear -except current_dir;
    clc;
    close all;
    
    % Create required directories
    required_dirs = {'Data', 'Data/Inputs', 'Data/Outputs', 'Models', 'Figures', 'ml_dataset'};
    for i = 1:length(required_dirs)
        if ~exist(required_dirs{i}, 'dir')
            mkdir(required_dirs{i});
            fprintf('Created directory: %s\n', required_dirs{i});
        end
    end
    
    %% Test Step 1: DT Setup
    fprintf('\n=== Testing Step 1: DT_setup.m ===\n');
    try
        [DT, DT_options] = DT_setup();
        fprintf('✓ DT_setup.m executed successfully\n');
        fprintf('  Digital twin model created with %d nodes and %d elements\n', ...
            length(DT.nodes), size(DT.elements, 1));
    catch ME
        fprintf('✗ Error in DT_setup.m: %s\n', ME.message);
    end
    
    %% Test Step 2: Limit States
    fprintf('\n=== Testing Step 2: limit_states.m ===\n');
    try
        [limit_state_funcs, ls_options] = limit_states(DT);
        fprintf('✓ limit_states.m executed successfully\n');
        fprintf('  Defined %d limit state functions\n', length(fieldnames(limit_state_funcs)));
    catch ME
        fprintf('✗ Error in limit_states.m: %s\n', ME.message);
    end
    
    %% Test Step 3: Compute Reliability
    fprintf('\n=== Testing Step 3: compute_reliability.m ===\n');
    try
        [beta, pf, rel_options] = compute_reliability(DT, limit_state_funcs);
        fprintf('✓ compute_reliability.m executed successfully\n');
        fprintf('  Reliability index (beta): %.4f\n', beta);
        fprintf('  Probability of failure: %.6e\n', pf);
    catch ME
        fprintf('✗ Error in compute_reliability.m: %s\n', ME.message);
    end
    
    %% Test Step 4: Compute Robustness
    fprintf('\n=== Testing Step 4: compute_robustness.m ===\n');
    try
        [RI, rob_options] = compute_robustness(DT, limit_state_funcs, beta);
        fprintf('✓ compute_robustness.m executed successfully\n');
        fprintf('  Robustness index (RI): %.4f\n', RI);
    catch ME
        fprintf('✗ Error in compute_robustness.m: %s\n', ME.message);
    end
    
    %% Test Step 5: RRI Calculation
    fprintf('\n=== Testing Step 5: RRI.m ===\n');
    try
        [RRI_value, RRI_options] = RRI(beta, RI);
        fprintf('✓ RRI.m executed successfully\n');
        fprintf('  RRI value: %.4f\n', RRI_value);
    catch ME
        fprintf('✗ Error in RRI.m: %s\n', ME.message);
    end
    
    %% Test Step 6: Generate Dataset
    fprintf('\n=== Testing Step 6: generate_dataset.m ===\n');
    try
        % Use a small number of samples for testing
        num_samples = 10;
        dataset_path = fullfile(current_dir, 'Data/Outputs/test_dataset.mat');
        
        dataset = generate_dataset(num_samples, dataset_path);
        fprintf('✓ generate_dataset.m executed successfully\n');
        fprintf('  Generated dataset with %d samples\n', num_samples);
        fprintf('  Mean RRI: %.4f\n', dataset.stats.mean_RRI);
    catch ME
        fprintf('✗ Error in generate_dataset.m: %s\n', ME.message);
    end
    
    %% Test Step 7: Export for Deep Learning
    fprintf('\n=== Testing Step 7: export_for_dl.m ===\n');
    try
        ml_dataset_dir = fullfile(current_dir, 'ml_dataset');
        [X_train, X_val, X_test, Y_train, Y_val, Y_test, metadata] = ...
            export_for_dl(dataset_path, ml_dataset_dir, [0.7, 0.15, 0.15]);
        
        fprintf('✓ export_for_dl.m executed successfully\n');
        fprintf('  Train samples: %d\n', size(X_train, 1));
        fprintf('  Validation samples: %d\n', size(X_val, 1));
        fprintf('  Test samples: %d\n', size(X_test, 1));
    catch ME
        fprintf('✗ Error in export_for_dl.m: %s\n', ME.message);
    end
    
    %% Test Step 8: Train Rapid-RRI-Net
    fprintf('\n=== Testing Step 8: train_rapid_rri_net.m ===\n');
    try
        % Set up model options with minimal epochs for testing
        model_options = struct();
        model_options.epochs = 3; % Just a few epochs for testing
        model_options.batch_size = 2;
        model_options.learning_rate = 0.001;
        model_options.patience = 2;
        model_options.model_save_path = fullfile(current_dir, 'Models/test_model.mat');
        
        % Check if we have enough data to train
        if exist('X_train', 'var') && size(X_train, 1) > 1
            [model, history] = train_rapid_rri_net(ml_dataset_dir, model_options);
            fprintf('✓ train_rapid_rri_net.m executed successfully\n');
            fprintf('  Final training loss: %.6f\n', history.loss(end));
        else
            fprintf('⚠ train_rapid_rri_net.m test skipped (not enough training data)\n');
        end
    catch ME
        fprintf('✗ Error in train_rapid_rri_net.m: %s\n', ME.message);
    end
    
    %% Test Step 9: Evaluate Model
    fprintf('\n=== Testing Step 9: evaluate_model.m ===\n');
    try
        % Try to use the model we just trained, or load a pre-existing one
        if ~exist('model', 'var') || ~exist(model_options.model_save_path, 'file')
            % Look for any model file
            model_files = dir(fullfile(current_dir, 'Models/*.mat'));
            if isempty(model_files)
                model_files = dir(fullfile(current_dir, 'Data/Outputs/*.mat'));
            end
            
            model_found = false;
            for i = 1:length(model_files)
                try
                    model_data = load(fullfile(model_files(i).folder, model_files(i).name));
                    if isfield(model_data, 'model')
                        model = model_data.model;
                        model_found = true;
                        fprintf('  Loaded model from: %s\n', fullfile(model_files(i).folder, model_files(i).name));
                        break;
                    end
                catch
                    continue;
                end
            end
            
            if ~model_found
                fprintf('⚠ evaluate_model.m test skipped (no model available)\n');
                return;
            end
        end
        
        % Prepare test data and evaluation options
        test_data = struct();
        test_data.X_test = X_test;
        test_data.y_test = Y_test;
        if exist('history', 'var')
            test_data.history = history;
        end
        
        eval_options = struct();
        eval_options.metrics = {'MAE', 'RMSE', 'R2'};
        eval_options.noise_level = 0.05;
        eval_options.n_trials = 2; % Small number for testing
        eval_options.output_path = fullfile(current_dir, 'Data/Outputs/test_eval_results.mat');
        eval_options.plot_confusion = false; % Skip plotting to speed up test
        eval_options.plot_trajectory = false;
        
        results = evaluate_model(model_options.model_save_path, test_data, eval_options);
        fprintf('✓ evaluate_model.m executed successfully\n');
        fprintf('  MAE: %.6f\n', results.metrics.mae);
        fprintf('  RMSE: %.6f\n', results.metrics.rmse);
        fprintf('  R-squared: %.6f\n', results.metrics.r2);
    catch ME
        fprintf('✗ Error in evaluate_model.m: %s\n', ME.message);
    end
    
    %% Show Model Predictions
    fprintf('\n=== Model Predictions ===\n');
    try
        if exist('model', 'var') && exist('X_test', 'var') && exist('Y_test', 'var')
            % Make predictions on a few samples
            num_samples_to_show = min(5, size(X_test, 1));
            
            % Make predictions
            predictions = predict(model, X_test(1:num_samples_to_show,:));
            
            % Display results
            fprintf('Sample | Actual RRI | Predicted RRI | Error\n');
            fprintf('-------------------------------------------\n');
            for i = 1:num_samples_to_show
                fprintf('%6d | %9.4f | %12.4f | %5.4f\n', ...
                    i, Y_test(i), predictions(i), abs(Y_test(i) - predictions(i)));
            end
        else
            fprintf('⚠ Model prediction display skipped (model or test data not available)\n');
        end
    catch ME
        fprintf('✗ Error showing predictions: %s\n', ME.message);
    end
    
    %% Test Completed
    fprintf('\n=== Test Complete ===\n');
    fprintf('All RRI functions have been tested\n');
    
catch ME
    % Display error message for the overall script
    fprintf('\n=== ERROR OCCURRED ===\n');
    fprintf('Error message: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Error in: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    end
end