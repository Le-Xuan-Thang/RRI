%% RRI Model Predictions Test
% This script tests the RRI model predictions by loading an existing model
% and showing its predictions on test data.

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

%% Initialize
try
    % Reset paths and add required paths
    restoredefaultpath;
    rehash toolboxcache;
    
    % Get current directory
    current_dir = pwd;
    
    % Add necessary paths with full paths to avoid conflicts
    addpath(fullfile(current_dir, 'src'));
    addpath(fullfile(current_dir, 'Generate Acc'));
    addpath("/Applications/MATLAB_R2024b.app/toolbox_Stabil");
    
    % Clean workspace
    clear -except current_dir;
    clc;
    close all;
    
    fprintf('Reset MATLAB path and added required folders to path\n');
    
    %% Load Model and Data
    fprintf('\n=== Loading Model and Data ===\n');
    
    % Check if model exists
    model_path = fullfile(current_dir, 'Models/rapid_rri_net.mat');
    if ~exist(model_path, 'file')
        % Try to find any model file in the Models folder
        model_files = dir(fullfile(current_dir, 'Models/*.mat'));
        if ~isempty(model_files)
            model_path = fullfile(model_files(1).folder, model_files(1).name);
            fprintf('Using model: %s\n', model_path);
        else
            % Check Data/Outputs for model files
            model_files = dir(fullfile(current_dir, 'Data/Outputs/*.mat'));
            if ~isempty(model_files)
                for i = 1:length(model_files)
                    % Load each file to check if it contains a model
                    try
                        data = load(fullfile(model_files(i).folder, model_files(i).name));
                        if isfield(data, 'model')
                            model_path = fullfile(model_files(i).folder, model_files(i).name);
                            fprintf('Found model in: %s\n', model_path);
                            break;
                        end
                    catch
                        % Skip files that can't be loaded
                        continue;
                    end
                end
            end
        end
    end
    
    % Check if we found a model
    if exist(model_path, 'file')
        fprintf('Loading model from: %s\n', model_path);
        model_data = load(model_path);
        
        if isfield(model_data, 'model')
            model = model_data.model;
            fprintf('Model loaded successfully\n');
            
            % Load test data
            fprintf('\n=== Loading Test Data ===\n');
            test_data_found = false;
            
            % Try option 1: Load from ml_dataset CSV files
            test_features_path = fullfile(current_dir, 'ml_dataset/test_features.csv');
            test_targets_path = fullfile(current_dir, 'ml_dataset/test_targets.csv');
            
            if exist(test_features_path, 'file') && exist(test_targets_path, 'file')
                fprintf('Loading test data from CSV files...\n');
                X_test = readmatrix(test_features_path);
                Y_test = readmatrix(test_targets_path);
                test_data_found = true;
            end
            
            % Try option 2: Find test data in .mat files
            if ~test_data_found
                fprintf('Test data CSV files not found. Looking for data in .mat files...\n');
                data_files = dir(fullfile(current_dir, 'Data/Outputs/*.mat'));
                
                for i = 1:length(data_files)
                    try
                        data = load(fullfile(data_files(i).folder, data_files(i).name));
                        % Check for test data variables
                        if isfield(data, 'X_test') && isfield(data, 'Y_test')
                            X_test = data.X_test;
                            Y_test = data.Y_test;
                            test_data_found = true;
                            fprintf('Found test data in: %s\n', fullfile(data_files(i).folder, data_files(i).name));
                            break;
                        end
                        % Alternative: look for results from evaluation
                        if isfield(data, 'test_data')
                            if isfield(data.test_data, 'X_test') && isfield(data.test_data, 'y_test')
                                X_test = data.test_data.X_test;
                                Y_test = data.test_data.y_test;
                                test_data_found = true;
                                fprintf('Found test data in: %s\n', fullfile(data_files(i).folder, data_files(i).name));
                                break;
                            end
                        end
                    catch
                        % Skip files that can't be loaded
                        continue;
                    end
                end
            end
            
            % Try option 3: Create synthetic dataset if needed
            if ~test_data_found
                fprintf('No test data found. Creating a new dataset for testing...\n');
                
                % Generate a small dataset
                fprintf('Generating a small dataset using generate_dataset...\n');
                num_samples = 20;
                dataset_path = fullfile(current_dir, 'Data/Outputs/test_dataset.mat');
                dataset = generate_dataset(num_samples, dataset_path);
                
                % Export for deep learning
                fprintf('Exporting dataset for deep learning...\n');
                [X_train, X_val, X_test, Y_train, Y_val, Y_test] = ...
                    export_for_dl(dataset_path, 'ml_dataset', [0.6, 0.2, 0.2]);
                
                test_data_found = true;
                fprintf('Created new test dataset with %d samples\n', size(X_test, 1));
            end
            
            %% Make Predictions and Calculate Metrics
            if test_data_found
                fprintf('\n=== Making Predictions ===\n');
                
                % Make predictions on all test data
                all_predictions = predict(model, X_test);
                
                % Calculate performance metrics
                mae = mean(abs(Y_test - all_predictions));
                rmse = sqrt(mean((Y_test - all_predictions).^2));
                r2 = 1 - sum((Y_test - all_predictions).^2) / sum((Y_test - mean(Y_test)).^2);
                correlation = corr(Y_test, all_predictions);
                
                % Display results
                fprintf('\n=== Model Performance Metrics ===\n');
                fprintf('MAE: %.6f\n', mae);
                fprintf('RMSE: %.6f\n', rmse);
                fprintf('R-squared: %.6f\n', r2);
                fprintf('Correlation: %.6f\n', correlation);
                
                % Display sample predictions
                fprintf('\n=== Sample Predictions ===\n');
                fprintf('Sample | Actual RRI | Predicted RRI | Error\n');
                fprintf('-------------------------------------------\n');
                num_samples_to_show = min(10, size(X_test, 1));
                
                for i = 1:num_samples_to_show
                    fprintf('%6d | %9.4f | %12.4f | %5.4f\n', ...
                        i, Y_test(i), all_predictions(i), abs(Y_test(i) - all_predictions(i)));
                end
                
                % Create a scatter plot of predictions vs actual values
                figure('Name', 'RRI Predictions vs. Actual', 'Position', [100, 100, 800, 600]);
                scatter(Y_test, all_predictions, 50, 'filled', 'MarkerFaceAlpha', 0.6);
                hold on;
                
                % Add perfect prediction line
                min_val = min(min(Y_test), min(all_predictions));
                max_val = max(max(Y_test), max(all_predictions));
                plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
                
                % Add linear fit line
                p = polyfit(Y_test, all_predictions, 1);
                fit_x = linspace(min_val, max_val, 100);
                fit_y = polyval(p, fit_x);
                plot(fit_x, fit_y, 'g-', 'LineWidth', 1.5);
                
                % Add labels and title
                xlabel('Actual RRI', 'FontWeight', 'bold');
                ylabel('Predicted RRI', 'FontWeight', 'bold');
                title('Rapid-RRI-Net: Predictions vs. Actual Values', 'FontSize', 14, 'FontWeight', 'bold');
                legend({'Test Samples', 'Perfect Prediction', 'Linear Fit'}, 'Location', 'northwest');
                grid on;
                axis equal tight;
                
                % Add text annotations with metrics
                text_x = min_val + 0.1 * (max_val - min_val);
                text_y = max_val - 0.1 * (max_val - min_val);
                text_metrics = sprintf('MAE: %.4f\nRMSE: %.4f\nR²: %.4f\nCorr: %.4f', mae, rmse, r2, correlation);
                text(text_x, text_y, text_metrics, 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
                
                % Save figure
                if ~exist('Figures', 'dir')
                    mkdir('Figures');
                end
                saveas(gcf, 'Figures/test_predictions.svg');
                fprintf('\nPrediction plot saved to Figures/test_predictions.svg\n');
            else
                fprintf('ERROR: Could not find or create test data\n');
            end
        else
            fprintf('ERROR: Loaded file does not contain a model\n');
        end
    else
        fprintf('ERROR: No model file found\n');
    end
    
    fprintf('\n=== Test Completed ===\n');
catch ME
    fprintf('\n=== ERROR OCCURRED ===\n');
    fprintf('Error message: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Error in: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    end
    fprintf('Stack trace:\n');
    disp(ME.stack);
end