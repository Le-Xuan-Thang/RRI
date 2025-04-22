%% Reliability-Robustness Index (RRI) Calculator
% This script calculates the Reliability-Robustness Index (RRI) by combining
% results from both reliability and robustness analysis.

function [RRI, results] = RRI(model, params, limit_values, options)
    % Calculate the Reliability-Robustness Index (RRI)
    %
    % Inputs:
    %   model: Function handle to the digital twin model
    %   params: Struct with random parameter definitions
    %   limit_values: Struct with allowable limits for limit states
    %   options: Options for the calculation (weights, methods, etc.)
    %
    % Outputs:
    %   RRI: The calculated Reliability-Robustness Index
    %   results: Detailed results structure with reliability and robustness info
    
    % Default options
    if nargin < 4
        options = struct();
    end
    
    % Set default options
    if ~isfield(options, 'w1')
        options.w1 = 0.6;  % Weight for reliability
    end
    if ~isfield(options, 'w2')
        options.w2 = 0.4;  % Weight for robustness
    end
    if ~isfield(options, 'rel_method')
        options.rel_method = 'MCS';  % Method for reliability calculation
    end
    if ~isfield(options, 'rob_method')
        options.rob_method = 'MCS';  % Method for robustness calculation
    end
    if ~isfield(options, 'rel_samples')
        options.rel_samples = 1000;  % Number of samples for reliability
    end
    if ~isfield(options, 'rob_samples')
        options.rob_samples = 100;  % Number of samples for robustness
    end
    if ~isfield(options, 'save_path')
        options.save_path = 'Data/RRI_results.mat';  % Path to save results
    end
    
    % Create Data directory if it doesn't exist
    if ~exist('Data', 'dir')
        mkdir('Data');
    end
    
    % Display analysis info
    fprintf('RRI Analysis Starting\n');
    fprintf('  Reliability method: %s with %d samples\n', ...
           options.rel_method, options.rel_samples);
    fprintf('  Robustness method: %s with %d samples\n', ...
           options.rob_method, options.rob_samples);
    fprintf('  Weights: Reliability (%.2f), Robustness (%.2f)\n', ...
           options.w1, options.w2);
    
    % Compute reliability index
    rel_options = struct('method', options.rel_method, ...
                       'n_samples', options.rel_samples);
    
    fprintf('\nComputing Reliability Index...\n');
    [beta, pf, g_samples] = compute_reliability(model, params, limit_values, rel_options);
    fprintf('  Reliability Index (β): %.4f\n', beta);
    fprintf('  Probability of Failure (Pf): %.4e\n', pf);
    
    % Compute robustness index
    rob_options = struct('method', options.rob_method, ...
                       'n_samples', options.rob_samples, ...
                       'w1', 0.5, ...
                       'w2', 0.5);
    
    fprintf('\nComputing Robustness Index...\n');
    [RI, RRI_rob, rob_results] = compute_robustness(model, params, limit_values, rob_options);
    fprintf('  Robustness Index (RI): %.4f\n', RI);
    
    % Normalize reliability index (assuming typical range is 0-5)
    beta_norm = min(max(beta / 5, 0), 1);
    
    % Calculate RRI as weighted sum
    RRI = options.w1 * beta_norm + options.w2 * RI;
    fprintf('\nReliability-Robustness Index (RRI): %.4f\n', RRI);
    
    % Create results structure
    results = struct();
    results.reliability = struct('beta', beta, 'pf', pf, 'samples', g_samples);
    results.robustness = struct('RI', RI, 'RRI_rob', RRI_rob, 'results', rob_results);
    results.RRI = RRI;
    results.options = options;
    
    % Create visualization
    visualize_results(results);
    
    % Save results if path provided
    if ~isempty(options.save_path)
        fprintf('\nSaving results to %s\n', options.save_path);
        save(options.save_path, 'results');
    end
end

function visualize_results(results)
    % Create visualization of RRI analysis results
    
    % Create Figures directory if it doesn't exist
    if ~exist('Figures', 'dir')
        mkdir('Figures');
    end
    
    % Create figure
    figure('Position', [100, 100, 1200, 400]);
    
    % 1. Reliability histogram
    subplot(1, 3, 1);
    histogram(results.reliability.samples, 30);
    xlabel('Limit State Value (g)');
    ylabel('Frequency');
    title(sprintf('Reliability: β = %.2f (Pf = %.1e)', ...
         results.reliability.beta, results.reliability.pf));
    grid on;
    % Add line at g = 0
    xlims = xlim;
    hold on;
    plot([0, 0], ylim, 'r--', 'LineWidth', 2);
    xlim(xlims);
    hold off;
    
    % 2. Phase portrait
    subplot(1, 3, 2);
    scatter(results.robustness.results.performance_original, ...
           results.robustness.results.performance_damaged, 30, 'filled');
    hold on;
    % Add 45-degree line
    plot([0, 1], [0, 1], 'k--');
    xlabel('Original Performance');
    ylabel('Performance After Damage');
    title(sprintf('Phase Portrait (RI = %.2f)', results.robustness.RI));
    grid on;
    axis([0 1 0 1]);
    hold off;
    
    % 3. RRI visualization
    subplot(1, 3, 3);
    reliability_norm = min(results.reliability.beta / 5, 1);
    bar([reliability_norm, results.robustness.RI, results.RRI]);
    xticklabels({'Reliability', 'Robustness', 'RRI'});
    ylabel('Index Value');
    title('Reliability-Robustness Index (RRI)');
    ylim([0 1]);
    grid on;
    
    % Save figure
    saveas(gcf, 'Figures/RRI_analysis.png');
end