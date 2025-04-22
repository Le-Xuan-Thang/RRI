%% Example: RRI Model Evaluation
% This script demonstrates how to use the evaluate_model function
% -----------------------------------------------------------------------
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% -----------------------------------------------------------------------

% Add source directory to path
addpath('src');

%% Load or create test data
fprintf('Setting up test data...\n');

% Check if we have a model to evaluate
if exist(fullfile('Models', 'rri_model.mat'), 'file')
    model_path = fullfile('Models', 'rri_model.mat');
    fprintf('Using existing model: %s\n', model_path);
else
    % Create a simple mock model for demonstration
    fprintf('Creating a simple mock model for demonstration...\n');
    
    % Generate synthetic data
    n_samples = 100;
    n_features = 5;
    X = randn(n_samples, n_features);
    
    % True coefficients
    true_coef = [0.5; -0.2; 0.7; 0.1; -0.4];
    
    % Generate target values with some noise
    y = X * true_coef + 0.1 * randn(n_samples, 1);
    
    % Create a simple linear model
    model = fitlm(X, y);
    
    % Save the model
    if ~exist('Models', 'dir')
        mkdir('Models');
    end
    model_path = fullfile('Models', 'rri_model.mat');
    save(model_path, 'model');
    fprintf('Mock model saved to %s\n', model_path);
end

% Prepare test data
n_test = 30;
X_test = randn(n_test, 5);
true_coef = [0.5; -0.2; 0.7; 0.1; -0.4];
y_test = X_test * true_coef + 0.1 * randn(n_test, 1);

% Create a training history structure for trajectory plotting
history = struct();
history.epoch = 1:20;
history.loss = exp(-0.2 * (1:20)) + 0.1 * rand(1, 20);
history.val_loss = history.loss + 0.05 * randn(1, 20);
history.mae = exp(-0.15 * (1:20)) + 0.05 * rand(1, 20);
history.val_mae = history.mae + 0.03 * randn(1, 20);
history.accuracy = 1 - exp(-0.1 * (1:20)) + 0.02 * rand(1, 20);
history.val_accuracy = history.accuracy - 0.04 * rand(1, 20);
history.rmse = sqrt(history.loss);
history.val_rmse = sqrt(history.val_loss);

% Create test data structure
test_data = struct();
test_data.X_test = X_test;
test_data.y_test = y_test;
test_data.history = history;

%% Set evaluation options
options = struct();
options.metrics = {'MAE', 'RMSE', 'R2', 'Accuracy', 'Loss'};
options.noise_level = 0.1;  % Test robustness with 10% noise
options.output_path = fullfile('Data', 'Outputs', 'evaluation_results.mat');
options.plot_confusion = true;
options.plot_trajectory = true;

%% Run evaluation
fprintf('\nEvaluating model...\n');
results = evaluate_model(model_path, test_data, options);

%% Display results summary
fprintf('\nEvaluation Results Summary:\n');
fprintf('---------------------------\n');

% Display all metrics
if isfield(results, 'metrics')
    fprintf('Performance Metrics:\n');
    metric_fields = fieldnames(results.metrics);
    for i = 1:length(metric_fields)
        metric_name = metric_fields{i};
        metric_value = results.metrics.(metric_name);
        fprintf('  %s: %.4f\n', upper(metric_name), metric_value);
    end
end

% Display speed results
if isfield(results, 'speed') && isfield(results.speed, 'mean')
    fprintf('\nInference Speed:\n');
    fprintf('  Mean: %.2f samples/second\n', results.speed.mean);
    fprintf('  Std:  %.2f samples/second\n', results.speed.std);
    fprintf('  Min:  %.2f samples/second\n', results.speed.min);
    fprintf('  Max:  %.2f samples/second\n', results.speed.max);
end

% Display robustness results
if isfield(results, 'robustness')
    fprintf('\nRobustness to Noise (%.1f%% level):\n', options.noise_level * 100);
    robust_fields = fieldnames(results.robustness);
    for i = 1:length(robust_fields)
        field = robust_fields{i};
        if ~strcmp(field, 'predictions') && ~endsWith(field, '_degradation')
            fprintf('  %s with noise: %.4f', upper(field), results.robustness.(field));
            if isfield(results.robustness, [field '_degradation'])
                fprintf(' (degradation: %.4f)', results.robustness.([field '_degradation']));
            end
            fprintf('\n');
        end
    end
end

fprintf('\nEvaluation completed!\n');
fprintf('Results saved to %s\n', options.output_path);
fprintf('Visualizations saved to Figures directory\n');