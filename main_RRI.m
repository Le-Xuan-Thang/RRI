%% Main RRI Calculation Script
% This script demonstrates the complete RRI workflow (steps 1-4)
% for the Cua Rao Bridge using the STABIL framework

% Reset MATLAB path to default before adding new paths
restoredefaultpath;
rehash toolboxcache;

% Get current directory
current_dir = pwd;
%% Generate random samples for parameter
% Add necessary paths with full paths to avoid conflicts
% Check if the path exists before adding
load_packages(current_dir);
%% Step 1: Digital Twin Setup
disp('====== Step 1: Digital Twin Setup ======');
[model, params] = DT_setup();

%% Step 2: Define Limit Values for Limit State Functions
disp('====== Step 2: Define Limit States ======');
limit_values = struct();
limit_values.stress_allow = 355e6;      % Steel yield stress (Pa)
limit_values.disp_allow = 0.05;         % Maximum allowable displacement (m)
limit_values.rotation_allow = 0.01;     % Maximum allowable rotation (rad)
limit_values.freq_min = 0.5;            % Minimum required frequency (Hz)

%% Step 3: Compute Reliability Index
disp('====== Step 3: Compute Reliability Index ======');
% Set options for reliability analysis
options = struct();
options.method = 'MCS';        % Monte Carlo Simulation
options.n_samples = 10;       % Number of samples (use small number for testing)

% Compute reliability index
[beta, Pf, rel_results] = compute_reliability(model, params, limit_values, options);

% Display results
fprintf('Reliability Index (β): %.4f\n', beta);
fprintf('Probability of Failure (Pf): %.6e\n', Pf);

%% Step 4: Compute Robustness Index and RRI
disp('====== Step 4: Compute Robustness Index and RRI ======');
% Set options for robustness analysis
rob_options = struct();
rob_options.method = 'MCS';    % Monte Carlo Simulation
rob_options.n_samples = 100;   % Number of samples (use small number for testing)
rob_options.w1 = 0.6;          % Weight for reliability in RRI
rob_options.w2 = 0.4;          % Weight for robustness in RRI
rob_options.damage_scenarios = {'reduce_stiffness'};  % Only one scenario for testing

% Compute robustness index and RRI
[RI, RRI, rob_results] = compute_robustness(model, params, limit_values, rob_options);

% Display results
fprintf('Robustness Index (RI): %.4f\n', RI);
fprintf('Reliability-Robustness Index (RRI): %.4f\n', RRI);

%% Save Results
if ~exist('Data', 'dir')
    mkdir('Data');
end
save('Data/RRI_results.mat', 'beta', 'Pf', 'RI', 'RRI', 'rel_results', 'rob_results');
disp('Results saved to Data/RRI_results.mat');

%% Plot Results (Optional)
% Create figure for reliability analysis
figure;
subplot(2,1,1);
histogram(rel_results.g_values, 20);
title('Limit State Function Values');
xlabel('g(X)');
ylabel('Frequency');
grid on;

% Add vertical line at g = 0
hold on;
line([0 0], ylim, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
hold off;

% Create figure for RRI
subplot(2,1,2);
bar([beta/5, RI, RRI]);
set(gca, 'XTickLabel', {'β/5', 'RI', 'RRI'});
ylim([0 1]);
title('Reliability and Robustness Indices');
grid on;

% Save figure
if ~exist('Figures', 'dir')
    mkdir('Figures');
end
saveas(gcf, 'Figures/RRI_analysis.svg');
disp('Plots saved to Figures/RRI_analysis.svg');