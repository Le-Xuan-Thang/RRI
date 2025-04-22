%% Digital Twin Setup & Calibration
% This script sets up the Digital Twin (DT) for the RRI project
% It uses the existing FE model from CuaRaoBridge_main.m and adds random parameters

function [model, params] = DT_setup()
    % Initialize parameter structure
    params = struct();
    
    %% Define random parameters with their distributions
    % E: Young's modulus (Log-Normal)
    params.E_mean = 3.5e10;    % Mean value [kN/m^2]
    params.E_std = 0.5e10;     % Standard deviation
    params.E_cov = 0.15;       % Coefficient of variation
    params.E_dist = 'lognormal';
    
    % ρ: Density (Normal)
    params.rho_mean = 2500;    % Mean value [kg/m^3]
    params.rho_std = 125;      % Standard deviation
    params.rho_dist = 'normal';
    
    % RH: Relative Humidity (GEV)
    params.RH_k = 0.2;         % Shape parameter
    params.RH_sigma = 10;      % Scale parameter
    params.RH_mu = 70;         % Location parameter
    params.RH_dist = 'gev';
    
    % T: Temperature (GEV)
    params.T_k = -0.1;         % Shape parameter
    params.T_sigma = 5;        % Scale parameter
    params.T_mu = 20;          % Location parameter
    params.T_dist = 'gev';
    
    % qw: Wind load (Gumbel)
    params.qw_mu = 0.5e3;      % Location parameter [N/m^2]
    params.qw_beta = 0.1e3;    % Scale parameter
    params.qw_dist = 'gumbel';
    
    % δdeg: Degradation factor (Gamma)
    params.deg_a = 2;          % Shape parameter
    params.deg_b = 0.05;       % Scale parameter
    params.deg_dist = 'gamma';
    
    % Load the existing FE model (add path if needed)
    addpath('Generate Acc');
    
    % Create a model wrapper (assuming CuaRaoBridge_main accepts these parameters)
    model = @(params) run_model(params);
    
    % Output
    disp('Digital Twin setup complete with random parameters defined');
end

function results = run_model(params_sample)
    % This function runs the bridge FE model with specific parameter values
    
    % Extract parameters for this simulation
    E = params_sample.E;
    rho = params_sample.rho;
    RH = params_sample.RH;
    T = params_sample.T;
    qw = params_sample.qw;
    deg = params_sample.deg;
    
    % Display the parameters used for this run
    fprintf('Running model with parameters:\n');
    fprintf('E: %.2e, ρ: %.2f, RH: %.2f, T: %.2f, qw: %.2f, deg: %.2f\n', ...
            E, rho, RH, T, qw, deg);

    % Store current directory to return to it after running the model
    currentDir = pwd;
    
    % Change to the directory with the bridge model
    addpath('Generate Acc');
    
    % Run the bridge model while adjusting the material properties
    % We'll need to modify material properties based on our random parameters
    try
        % Load nodes and elements
        run('Generate Acc/CuaRao_Nodes.m');
        run('Generate Acc/CuaRao_Elements.m');
        
        % Run the FE model
        CuaRaoBridge_main

        fprintf("Old material properties:\n");
        fprintf("E: %.2e, ρ: %.2f\n", E, rho);

        % Apply wind load (qw) based on random parameter
        % The load is applied perpendicular to the bridge deck
        % Simplified: assume uniform pressure on exposed surfaces
        
        % Define loading vector (representative, would need adjusting)
        % F = zeros(size(K,1), 1);
        % wind_nodes = []; % Define nodes where wind load is applied
        % for i = 1:length(wind_nodes)
        %     dof_indices = []; % Define DOF indices for these nodes
        %     F(dof_indices) = F(dof_indices) + qw * exposed_area;
        % end
        
        % Static analysis (displacement and stress)
        % U = K\F;
        % displacements = zeros(size(Nodes,1)*6, 1);
        % displacements(DOF) = U;
        
        % Store results in struct
        results = struct();
        results.frequencies = f0;
        % results.displacements = displacements;
        % results.phi = phi; % Mode shapes
        
        % Calculate maximum rotation and displacement
        % For demonstration, use placeholders
        % In practice, this would be calculated from the FE results
        results.displacements = zeros(10, 1) * (1 + 0.1*RH/100); % Placeholder affected by RH
        results.rotations = zeros(10, 1) * (1 + 0.05*T/30);      % Placeholder affected by T
        results.stresses = zeros(10, 1) * (1 + 0.2*qw/1000);     % Placeholder affected by wind load
        
    catch ME
        % If there's an error, return debugging information
        disp(['Error in run_model: ', ME.message]);
        results = struct();
        results.frequencies = [1.5, 2.3, 3.1]; % Example placeholder
        results.displacements = zeros(10, 1);  % Example placeholder
        results.stresses = zeros(10, 1);       % Example placeholder
        results.rotations = zeros(10, 1);      % Example placeholder
        results.error = ME.message;
    end
    
    % Return to original directory
    cd(currentDir);
end

%% Model Calibration
% Optional: Model calibration function / This one use in the case we have
% measurement results
function params_calibrated = calibrate_model(params, measured_freqs, measured_modes)
    % This function would calibrate E and ρ to match measured frequencies/modes
    % Implement if you have real measurement data
    
    % Simple placeholder for demonstration
    params_calibrated = params;
    
    % Calibration would typically use an optimization algorithm
    % to minimize the error between simulated and measured response
end