%% Digital Twin Setup & Calibration
% This script sets up the Digital Twin (DT) for the RRI project
% It uses the existing FE model from CuaRaoBridge_main.m and adds random parameters

function [model, params] = DT_setup()
    % Initialize parameter structure
    params = struct();
    
    % Define random parameters with their distributions
    % E: Young's modulus (Log-Normal)
    params.E_mean = 2.0e11;    % Mean value [N/m2]
    params.E_cov = 0.15;       % Coefficient of variation
    params.E_dist = 'lognormal';
    
    % ρ: Density (Normal)
    params.rho_mean = 7800;    % Mean value [kg/m^3]
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
    model = @(params_sample) run_model(params_sample);
    
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
%     fprintf("Running model with:\n");
%     fprintf("  E: %.2e\n", E);
%     fprintf("  rho: %.2f\n", rho);
%     fprintf("  RH: %.2f\n", RH);
%     fprintf("  T: %.2f\n", T);
%     fprintf("  qw: %.2f\n", qw);
%     fprintf("  deg: %.2f\n", deg);
    
    % Store current directory to return to it after running the model
    currentDir = pwd;
    
    % Run the bridge model while adjusting the material properties
    % We'll need to modify material properties based on our random parameters
    try
        % Load the existing model
        run('Generate Acc/CuaRaoBridge_main.m')
        
        % 2. Inject random material updates ------------------------------------
        % Young’s modulus & density
        Materials(:,2) = E;                    % update E
        Materials(:,4) = rho;                  % update ρ (column 4 = ρ)
    
        % Simple degradation: reduce sectional area only (can be refined)
        Sections(:,2)  = Sections(:,2) .* (1 - deg);  % A := A·(1‑δ)

        % 3. Assemble global K once (faster when sampling) ---------------------
        [K,M] = asmkm(Nodes, Elements, Types, Sections, Materials, DOF);

        % 4. Build distributed wind load (example 1.2, Stabil manual) ----------
        %    Apply uniform lateral pressure ±qw along +Y direction on *all*
        %    beam elements. Modify selector below if you want a subset.
        %    choose element
        E_eff = [linspace(1,8,8), linspace(31,38,8), linspace(39,46,8), linspace(17,23,7)];
        nElem  = size(E_eff,2);

        % qw [N/m²] realisation

        % Own weight
        DLoadsOwn=accel([0 0 9.81],Elements,Types,Sections,Materials);

        % Each row: [EltID n1X n1Y n1Z n2X n2Y n2Z]
        DLoadsWind = [Elements(E_eff,1), ...             % EltID
                  zeros(nElem,1),  qw*ones(nElem,1), zeros(nElem,1), ...
                  zeros(nElem,1),  qw*ones(nElem,1), zeros(nElem,1) ];

        DLoads=multdloads(DLoadsOwn,DLoadsWind);
        % Convert distributed loads → equivalent nodal forces
        P = elemloads(DLoads, Nodes, Elements, Types, DOF);
    
        % 5. Solve static problem (wind‑only) ----------------------------------
        U = K \ P;                    % nodal displacements

        % 6. Post‑process basic responses (extend as needed) -------------------
        % Element forces → stresses (global CS)
        % Compute forces
        [ForcesLCS,ForcesGCS]=elemforces(Nodes,Elements,Types,Sections,Materials,DOF,U,DLoads);
        % Load combinations
        % Safety factors
        gamma_own=1.35;
        gamma_wind=1.5;
        % Combination factors
        psi_wind=1;
        % Load combination (Ultimate Limit State, ULS)
%         U_ULS=gamma_own*U(:,1)+gamma_wind*psi_wind*U(:,2);
        Forces_ULS=gamma_own*ForcesLCS(:,:,1)+gamma_wind*psi_wind*ForcesLCS(:,:,2);
        DLoads_ULS(:,1)=DLoads(:,1,1);
        DLoads_ULS(:,2:7)=gamma_own*DLoads(:,2:7,1)+gamma_wind*psi_wind*DLoads(:,2:7,2);
%         figure;
%         plotstress('smomzt',Nodes,Elements,Types,Sections,Materials,Forces_ULS,DLoads_ULS)
        [Stress,loc,Ext] = se_beam('smax',Nodes,Elements,Types,Sections,...
                            Materials,Forces_ULS,DLoads_ULS);
        elemMax = cellfun(@(s) max(s), Stress);   % 1×nElemBeams
        % Eigenvalue problem
        nMode=12;
        [~,omega]=eigfem(K,M,nMode);
        f0 = omega/2/pi;

        % Collect outputs – minimal set for reliability / robustness routines
        results            = struct();
        results.displacements   = max(abs(U));                  % useful scalar
        results.stress     = max(elemMax);                       % element forces
        results.frequencies= f0;                           % modes from main file
        results.qw         = qw;                           % store actual wind
        
    catch ME
        % If there's an error, return debugging information
        disp(['Error in run_model: ', ME.message]);
        results = struct();
        results.frequencies = [1.5, 2.3, 3.1]; % Example placeholder
        results.displacements = zeros(10, 1);  % Example placeholder
        results.stresses = zeros(10, 1);       % Example placeholder
%         results.rotations = zeros(10, 1);      % Example placeholder
        results.error = ME.message;
    end
    
    % Return to original directory
    cd(currentDir);
end

% Optional: Model calibration function
function params_calibrated = calibrate_model(params, measured_freqs, measured_modes)
    % This function would calibrate E and ρ to match measured frequencies/modes
    % Implement if you have real measurement data
    
    % Simple placeholder for demonstration
    params_calibrated = params;
    
    % Calibration would typically use an optimization algorithm
    % to minimize the error between simulated and measured response
end