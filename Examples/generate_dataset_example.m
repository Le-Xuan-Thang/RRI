%% Generate Dataset Example
% This example script demonstrates how to use the generate_dataset function
% to create a small dataset for the RRI project

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

%% Initialize
clear;
clc;
close all;

% Add the src directory to the MATLAB path to access functions
addpath('../src');

%% Set Parameters for Small Dataset Generation
% Define the number of samples to generate (using a small number for demonstration)
num_samples = 100;

% Define the save path for the dataset
save_path = '../Data/Inputs/example_dataset.mat';

fprintf('Generating a small dataset with %d samples for demonstration purposes...\n', num_samples);

%% Generate the Dataset
% Call generate_dataset function with the defined parameters
dataset = generate_dataset(num_samples, save_path);

%% Explore the Dataset
fprintf('\nExploring the generated dataset:\n');

% Display dataset structure
fprintf('Dataset structure contains following components:\n');
fprintf('- dataset.features: Model outputs (displacements, stresses, etc.)\n');
fprintf('- dataset.metadata: Random parameter values\n');
fprintf('- dataset.targets: Reliability and robustness indices\n');
fprintf('- dataset.stats: Statistical information about the dataset\n\n');

% Display dataset statistics
fprintf('Dataset Statistics:\n');
fprintf('Mean RRI: %.4f\n', dataset.stats.mean_RRI);
fprintf('Standard Deviation of RRI: %.4f\n', dataset.stats.std_RRI);
fprintf('Min RRI: %.4f\n', dataset.stats.min_RRI);
fprintf('Max RRI: %.4f\n', dataset.stats.max_RRI);
fprintf('Valid samples: %d of %d\n', dataset.stats.valid_samples, num_samples);
fprintf('Generation time: %.2f seconds (%.2f samples/second)\n\n', ...
    dataset.stats.generation_time, num_samples/dataset.stats.generation_time);

%% Visualize Dataset Features
fprintf('Creating visualizations of the dataset...\n');

% Figure 1: Distribution of RRI values
figure('Name', 'RRI Distribution', 'Position', [100, 100, 800, 600]);
histogram(dataset.targets.RRI, 20, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'none');
hold on;
xline(dataset.stats.mean_RRI, 'r-', 'LineWidth', 2, 'DisplayName', 'Mean');
xline(dataset.stats.mean_RRI - dataset.stats.std_RRI, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Mean ± Std');
xline(dataset.stats.mean_RRI + dataset.stats.std_RRI, 'r--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
hold off;
xlabel('RRI Value', 'FontWeight', 'bold');
ylabel('Frequency', 'FontWeight', 'bold');
title('Distribution of Reliability-Robustness Index (RRI)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% Figure 2: Parameter Influence on RRI
figure('Name', 'Parameter Influence', 'Position', [100, 100, 1200, 800]);

% Create subplots for each parameter vs RRI
subplot(2, 3, 1);
scatter(dataset.metadata.E, dataset.targets.RRI, 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Young''s Modulus (E)', 'FontWeight', 'bold');
ylabel('RRI', 'FontWeight', 'bold');
title('Influence of E on RRI', 'FontSize', 12);
grid on;

subplot(2, 3, 2);
scatter(dataset.metadata.rho, dataset.targets.RRI, 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Density (ρ)', 'FontWeight', 'bold');
ylabel('RRI', 'FontWeight', 'bold');
title('Influence of ρ on RRI', 'FontSize', 12);
grid on;

subplot(2, 3, 3);
scatter(dataset.metadata.RH, dataset.targets.RRI, 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Relative Humidity (RH)', 'FontWeight', 'bold');
ylabel('RRI', 'FontWeight', 'bold');
title('Influence of RH on RRI', 'FontSize', 12);
grid on;

subplot(2, 3, 4);
scatter(dataset.metadata.T, dataset.targets.RRI, 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Temperature (T)', 'FontWeight', 'bold');
ylabel('RRI', 'FontWeight', 'bold');
title('Influence of T on RRI', 'FontSize', 12);
grid on;

subplot(2, 3, 5);
scatter(dataset.metadata.qw, dataset.targets.RRI, 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Wind Load (qw)', 'FontWeight', 'bold');
ylabel('RRI', 'FontWeight', 'bold');
title('Influence of qw on RRI', 'FontSize', 12);
grid on;

subplot(2, 3, 6);
scatter(dataset.metadata.deg, dataset.targets.RRI, 40, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Degradation Factor (δdeg)', 'FontWeight', 'bold');
ylabel('RRI', 'FontWeight', 'bold');
title('Influence of δdeg on RRI', 'FontSize', 12);
grid on;

sgtitle('Relationship Between Random Parameters and RRI', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 3: Correlation between reliability (beta) and robustness (RI)
figure('Name', 'Reliability vs Robustness', 'Position', [100, 100, 800, 600]);
scatter(dataset.targets.beta, dataset.targets.RI, 60, dataset.targets.RRI, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel('Reliability Index (β)', 'FontWeight', 'bold');
ylabel('Robustness Index (RI)', 'FontWeight', 'bold');
title('Relationship Between Reliability and Robustness', 'FontSize', 14, 'FontWeight', 'bold');
colorbar('TickLabelInterpreter', 'latex', 'Label', 'RRI Value');
grid on;
box on;

fprintf('Example completed successfully. Three figures were generated to visualize the dataset.\n');