%% Compute Robustness Index (RI) & RRI
% This script computes the Robustness Index (RI) and combines it with
% Reliability Index (β) to get the Reliability-Robustness Index (RRI)

function [RI, RRI, results] = compute_robustness(model, params, limit_values, options)
    % Compute robustness index (RI) for the structural system and combine
    % it with reliability index (β) to get RRI
    %
    % Inputs:
    %   model: function handle to the FE model wrapper
    %   params: struct with random parameter definitions
    %   limit_values: struct with allowable limits for limit states
    %   options: struct with analysis options
    %
    % Outputs:
    %   RI: robustness index
    %   RRI: reliability-robustness index
    %   results: additional results and diagnostics
    
    % Set default options if not provided
    if nargin < 4
        options = struct();
    end
    
    % Default options
    if ~isfield(options, 'method')
        options.method = 'FORM';  % 'FORM', 'SubSim', or 'MCS'
    end
    if ~isfield(options, 'n_samples')
        options.n_samples = 1000;  % Number of samples for Monte Carlo methods
    end
    if ~isfield(options, 'w1')
        options.w1 = 0.6;  % Weight for reliability in RRI
    end
    if ~isfield(options, 'w2')
        options.w2 = 0.4;  % Weight for robustness in RRI
    end
    if ~isfield(options, 'damage_scenarios')
        % Default damage scenarios to test
        options.damage_scenarios = {
            'remove_element',  % Remove a critical element
            'reduce_stiffness',  % Reduce stiffness by 30%
            'foundation_settlement'  % Apply foundation settlement
        };
    end
    
    % Initialize results structure
    results = struct();
    results.options = options;
    results.beta_intact = 0;
    results.Pf_intact = 0;
    results.beta_damaged = zeros(length(options.damage_scenarios), 1);
    results.Pf_damaged = zeros(length(options.damage_scenarios), 1);
    results.damage_scenarios = options.damage_scenarios;
    
    % Step 1: Compute reliability index for intact structure
    fprintf('Computing reliability index for intact structure...\n');
    [beta_intact, Pf_intact, results_intact] = compute_reliability(model, params, limit_values, options);
    results.beta_intact = beta_intact;
    results.Pf_intact = Pf_intact;
    results.reliability_results_intact = results_intact;
    
    % Step 2: Compute reliability index for each damage scenario
    fprintf('Computing reliability indices for damage scenarios...\n');
    for i = 1:length(options.damage_scenarios)
        % Create damaged model
        scenario = options.damage_scenarios{i};
        fprintf('Scenario %d: %s\n', i, scenario);
        
        % Create a wrapper for the damaged model
        damaged_model = @(params_sample) apply_damage(model, params_sample, scenario);
        
        % Compute reliability for damaged model
        [beta_d, Pf_d, results_d] = compute_reliability(damaged_model, params, limit_values, options);
        
        % Store results
        results.beta_damaged(i) = beta_d;
        results.Pf_damaged(i) = Pf_d;
        results.reliability_results_damaged{i} = results_d;
    end
    
    % Step 3: Compute Robustness Index (RI)
    % RI = 1 - (Pf,d / Pf,0)
    % Use the worst-case damage scenario (maximum Pf)
    [Pf_worst, worst_idx] = max(results.Pf_damaged);
    
    % Check for division by zero
    if Pf_intact > 0
        RI = 1 - (Pf_worst / Pf_intact);
    else
        % If intact structure has no failures, use a small value
        RI = 1 - Pf_worst / (1/options.n_samples);
        fprintf('Warning: Intact structure has zero failure probability. Using approximation.\n');
    end
    
    % Ensure RI is between 0 and 1
    RI = max(0, min(1, RI));
    
    % Step 4: Compute Reliability-Robustness Index (RRI)
    % RRI = w1*β + w2*RI
    % Normalize beta to [0,1] range for combination
    beta_norm = min(1, beta_intact / 5);  % Assuming beta > 5 is "very safe"
    RRI = options.w1 * beta_norm + options.w2 * RI;
    
    % Store combined results
    results.RI = RI;
    results.RRI = RRI;
    results.worst_scenario = options.damage_scenarios{worst_idx};
    results.beta_norm = beta_norm;
    
    % Display results
    fprintf('\nRobustness Analysis Results:\n');
    fprintf('Reliability Index (β): %.4f\n', beta_intact);
    fprintf('Probability of Failure (Pf): %.6e\n', Pf_intact);
    fprintf('Robustness Index (RI): %.4f\n', RI);
    fprintf('Reliability-Robustness Index (RRI): %.4f\n', RRI);
    fprintf('Worst-case damage scenario: %s\n', results.worst_scenario);
end

% Helper function: Apply damage to the model
function results = apply_damage(model, params_sample, scenario)
    % Apply damage to the model based on the scenario
    switch scenario
        case 'remove_element'
            % Simulate removal of a critical element
            % In practice, you would modify the FE model directly
            results = apply_element_removal(model, params_sample);
            
        case 'reduce_stiffness'
            % Simulate reduction in stiffness
            % Typically reduce E by a factor
            params_reduced = params_sample;
            params_reduced.E = params_sample.E * 0.7;  % 30% reduction
            results = model(params_reduced);
            
        case 'foundation_settlement'
            % Simulate foundation settlement
            % In practice, you would modify boundary conditions
            results = apply_settlement(model, params_sample);
            
        otherwise
            error('Unknown damage scenario: %s', scenario);
    end
end

% Helper function: Apply element removal
function results = apply_element_removal(model, params_sample)
    % Placeholder for implementing element removal
    % In practice, you would modify the FE model structure
    
    % Run original model
    results = model(params_sample);
    
    % Modify results to simulate element removal effect
    % This is just a placeholder - in practice, you would re-run the FE analysis
    if isfield(results, 'displacements')
        results.displacements = results.displacements * 1.5;  % Simplified effect
    end
    if isfield(results, 'stresses')
        results.stresses = results.stresses * 1.8;  % Simplified effect
    end
    if isfield(results, 'frequencies')
        results.frequencies = results.frequencies * 0.9;  % Simplified effect
    end
end

% Helper function: Apply settlement
function results = apply_settlement(model, params_sample)
    % Placeholder for implementing foundation settlement
    % In practice, you would modify the boundary conditions
    
    % Run original model
    results = model(params_sample);
    
    % Modify results to simulate settlement effect
    % This is just a placeholder - in practice, you would re-run the FE analysis
    if isfield(results, 'displacements')
        results.displacements = results.displacements * 1.3;  % Simplified effect
    end
    if isfield(results, 'stresses')
        results.stresses = results.stresses * 1.4;  % Simplified effect
    end
    if isfield(results, 'rotations')
        results.rotations = results.rotations * 1.2;  % Simplified effect
    end
end