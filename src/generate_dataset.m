%% Generate Dataset for Deep Learning
% This script generates a comprehensive dataset for training and evaluation
% of the Rapid-RRI-Net deep learning model (step 5 of the RRI workflow)

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

function dataset = generate_dataset(num_samples, save_path)
    % Generate a comprehensive dataset for deep learning
    %
    % Inputs:
    %   num_samples: Number of samples to generate (default: 10,000)
    %   save_path: Path to save the dataset (default: 'Data/Inputs/rri_dataset.mat')
    %
    % Outputs:
    %   dataset: Structure containing the generated dataset
    
    % Set defaults
    if nargin < 1
        num_samples = 10000;
    end
    if nargin < 2
        save_path = 'Data/Inputs/rri_dataset.mat';
    end
    
    % Ensure save directory exists
    save_dir = fileparts(save_path);
    if ~isfolder(save_dir)
        mkdir(save_dir);
    end
    
    fprintf('Generating RRI dataset with %d samples...\n', num_samples);
    
    % Initialize dataset structures
    dataset = struct();
    dataset.features = struct(); % FE response data
    dataset.metadata = struct(); % Parameter values
    dataset.targets = struct();  % RRI values
    
    % Display progress bar parameters
    progress_interval = max(1, floor(num_samples / 100));
    fprintf('Progress: [');
    progress_dots = 0;
    
    % Set up Digital Twin model and parameters
    try
        [model, params_base] = DT_setup();
    catch ME
        error('Error setting up Digital Twin: %s', ME.message);
    end
    
    % Set up limit values for limit states
    limit_values = struct();
    limit_values.stress_allow = 250e6;    % Allowable stress [Pa]
    limit_values.disp_allow = 0.05;       % Allowable displacement [m]
    limit_values.rotation_allow = 0.01;   % Allowable rotation [rad]
    limit_values.freq_min = 1.0;          % Minimum frequency [Hz]
    
    % Set up analysis options
    options = struct();
    options.method = 'MCS';               % Use Monte Carlo Simulation for speed
    options.n_samples = 100;              % Smaller sample size for dataset generation
    
    % Initialize data arrays
    % Features (model outputs)
    displacements = zeros(num_samples, 10); % Assuming 10 displacement values
    rotations = zeros(num_samples, 10);     % Assuming 10 rotation values
    stresses = zeros(num_samples, 10);      % Assuming 10 stress values
    frequencies = zeros(num_samples, 12);   % Assuming 12 frequency values (from model)
    
    % Metadata (random parameters)
    E_values = zeros(num_samples, 1);
    rho_values = zeros(num_samples, 1);
    RH_values = zeros(num_samples, 1);
    T_values = zeros(num_samples, 1);
    qw_values = zeros(num_samples, 1);
    deg_values = zeros(num_samples, 1);
    
    % Targets (RRI values)
    beta_values = zeros(num_samples, 1);
    RI_values = zeros(num_samples, 1);
    RRI_values = zeros(num_samples, 1);
    
    % Start timer
    tic;
    
    % Generate samples
    for i = 1:num_samples
        try
            % Generate random parameter sample
            params_sample = generate_random_sample(params_base);
            
            % Run model with these parameters
            model_results = model(params_sample);
            
            % Compute reliability and robustness (with reduced samples for speed)
            [beta, Pf, rel_results] = compute_reliability(model, params_sample, limit_values, options);
            [RI, RRI, rob_results] = compute_robustness(model, params_sample, limit_values, options);
            
            % Store parameters (metadata)
            E_values(i) = params_sample.E;
            rho_values(i) = params_sample.rho;
            RH_values(i) = params_sample.RH;
            T_values(i) = params_sample.T;
            qw_values(i) = params_sample.qw;
            deg_values(i) = params_sample.deg;
            
            % Store model responses (features)
            if isfield(model_results, 'displacements')
                displacements(i, :) = model_results.displacements(1:min(10, length(model_results.displacements)));
            end
            if isfield(model_results, 'rotations')
                rotations(i, :) = model_results.rotations(1:min(10, length(model_results.rotations)));
            end
            if isfield(model_results, 'stresses')
                stresses(i, :) = model_results.stresses(1:min(10, length(model_results.stresses)));
            end
            if isfield(model_results, 'frequencies')
                frequencies(i, :) = model_results.frequencies(1:min(12, length(model_results.frequencies)));
            end
            
            % Store results (targets)
            beta_values(i) = beta;
            RI_values(i) = RI;
            RRI_values(i) = RRI;
            
            % Update progress bar
            if mod(i, progress_interval) == 0 || i == num_samples
                new_dots = floor(i/num_samples * 50);
                if new_dots > progress_dots
                    fprintf('%s', repmat('.', 1, new_dots - progress_dots));
                    progress_dots = new_dots;
                end
            end
            
        catch ME
            warning('Error in sample %d: %s. Skipping.', i, ME.message);
            % Fill with NaN values to indicate invalid sample
            displacements(i, :) = NaN;
            rotations(i, :) = NaN;
            stresses(i, :) = NaN;
            frequencies(i, :) = NaN;
            E_values(i) = NaN;
            rho_values(i) = NaN;
            RH_values(i) = NaN;
            T_values(i) = NaN;
            qw_values(i) = NaN;
            deg_values(i) = NaN;
            beta_values(i) = NaN;
            RI_values(i) = NaN;
            RRI_values(i) = NaN;
        end
    end
    
    % Complete progress bar
    fprintf('] Done.\n');
    
    % Store data in the dataset structure
    % Features
    dataset.features.displacements = displacements;
    dataset.features.rotations = rotations;
    dataset.features.stresses = stresses;
    dataset.features.frequencies = frequencies;
    
    % Metadata
    dataset.metadata.E = E_values;
    dataset.metadata.rho = rho_values;
    dataset.metadata.RH = RH_values;
    dataset.metadata.T = T_values;
    dataset.metadata.qw = qw_values;
    dataset.metadata.deg = deg_values;
    
    % Targets
    dataset.targets.beta = beta_values;
    dataset.targets.RI = RI_values;
    dataset.targets.RRI = RRI_values;
    
    % Calculate statistics
    dataset.stats.mean_RRI = mean(RRI_values, 'omitnan');
    dataset.stats.std_RRI = std(RRI_values, 'omitnan');
    dataset.stats.min_RRI = min(RRI_values, [], 'omitnan');
    dataset.stats.max_RRI = max(RRI_values, [], 'omitnan');
    dataset.stats.valid_samples = sum(~isnan(RRI_values));
    
    % Calculate elapsed time
    elapsed_time = toc;
    dataset.stats.generation_time = elapsed_time;
    
    % Display statistics
    fprintf('Dataset generation complete.\n');
    fprintf('Total samples: %d (valid: %d)\n', num_samples, dataset.stats.valid_samples);
    fprintf('RRI statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n', ...
        dataset.stats.mean_RRI, dataset.stats.std_RRI, ...
        dataset.stats.min_RRI, dataset.stats.max_RRI);
    fprintf('Generation time: %.1f seconds (%.2f samples/second)\n', ...
        elapsed_time, num_samples/elapsed_time);
    
    % Save dataset
    try
        save(save_path, 'dataset', '-v7.3');
        fprintf('Dataset saved to %s\n', save_path);
    catch ME
        warning('Failed to save dataset: %s', ME.message);
    end
end

% Helper function to generate a random sample based on parameter distributions
function params_sample = generate_random_sample(params)
    % Generate a random sample for each parameter based on its distribution
    params_sample = struct();
    
    % E: Young's modulus (Log-Normal)
    if isfield(params, 'E_dist') && strcmp(params.E_dist, 'lognormal')
        mu_ln = log(params.E_mean / sqrt(1 + params.E_cov^2));
        sigma_ln = sqrt(log(1 + params.E_cov^2));
        params_sample.E = lognrnd(mu_ln, sigma_ln);
    else
        params_sample.E = params.E_mean;
    end
    
    % ρ: Density (Normal)
    if isfield(params, 'rho_dist') && strcmp(params.rho_dist, 'normal')
        params_sample.rho = normrnd(params.rho_mean, params.rho_std);
    else
        params_sample.rho = params.rho_mean;
    end
    
    % RH: Relative Humidity (GEV)
    if isfield(params, 'RH_dist') && strcmp(params.RH_dist, 'gev')
        try
            params_sample.RH = gevrnd(params.RH_k, params.RH_sigma, params.RH_mu);
        catch
            % Alternative approach if gevrnd isn't available
            u = rand();
            params_sample.RH = params.RH_mu - params.RH_sigma * log(-log(u));
        end
    else
        params_sample.RH = params.RH_mu;
    end
    
    % T: Temperature (GEV)
    if isfield(params, 'T_dist') && strcmp(params.T_dist, 'gev')
        try
            params_sample.T = gevrnd(params.T_k, params.T_sigma, params.T_mu);
        catch
            % Alternative approach if gevrnd isn't available
            u = rand();
            params_sample.T = params.T_mu - params.T_sigma * log(-log(u));
        end
    else
        params_sample.T = params.T_mu;
    end
    
    % qw: Wind load (Gumbel)
    if isfield(params, 'qw_dist') && strcmp(params.qw_dist, 'gumbel')
        % Gumbel distribution (Type I extreme value)
        u = rand(); % Uniform random number
        params_sample.qw = params.qw_mu - params.qw_beta * log(-log(u));
    else
        params_sample.qw = params.qw_mu;
    end
    
    % δdeg: Degradation factor (Gamma)
    if isfield(params, 'deg_dist') && strcmp(params.deg_dist, 'gamma')
        try
            params_sample.deg = gamrnd(params.deg_a, params.deg_b);
        catch
            % Alternative approach if gamrnd isn't available
            params_sample.deg = sum(rand(12,1))/12 * params.deg_a * params.deg_b * 2;
        end
    else
        params_sample.deg = params.deg_a * params.deg_b;
    end
end