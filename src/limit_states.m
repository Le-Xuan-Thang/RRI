%% Limit States & State Functions
% This script defines limit state functions for reliability analysis
% g(X) <= 0 indicates failure, g(X) > 0 indicates safety

function [g, state] = limit_states(results, limit_values)
    % Define limit state functions for the FE model results
    % g(X) = R - S format (Resistance - Stress)
    
    % Input:
    %   results: struct containing FE model results
    %   limit_values: struct containing allowable limits
    %
    % Output:
    %   g: vector of limit state function values
    %   state: struct with additional information
    
    % Initialize output
    g = [];
    state = struct();
    
    % Limit State 1: Strength-based (g1 = R - S)
    % Example: Maximum stress vs. allowable stress
    if isfield(results, 'stresses') && isfield(limit_values, 'stress_allow')
        stress_max = max(abs(results.stresses));
        g1 = limit_values.stress_allow - stress_max;
        g = [g; g1];
        state.stress_ratio = stress_max / limit_values.stress_allow;
    end
    
    % Limit State 2: Rotation-based (g2 = θmax - θallow)
    % Example: Maximum rotation vs. allowable rotation
    if isfield(results, 'rotations') && isfield(limit_values, 'rotation_allow')
        rotation_max = max(abs(results.rotations));
        g2 = limit_values.rotation_allow - rotation_max;
        g = [g; g2];
        state.rotation_ratio = rotation_max / limit_values.rotation_allow;
    end
    
    % Limit State 3: Displacement-based
    % Example: Maximum displacement vs. allowable displacement
    if isfield(results, 'displacements') && isfield(limit_values, 'disp_allow')
        disp_max = max(abs(results.displacements));
        g3 = limit_values.disp_allow - disp_max;
        g = [g; g3];
        state.disp_ratio = disp_max / limit_values.disp_allow;
    end
    
    % Limit State 4: Frequency-based
    % Example: Minimum frequency vs. required frequency
    if isfield(results, 'frequencies') && isfield(limit_values, 'freq_min')
        freq_min = min(results.frequencies);
        g4 = freq_min - limit_values.freq_min;
        g = [g; g4];
        state.freq_ratio = freq_min / limit_values.freq_min;
    end
    
    % Selective MAX operation (most critical limit state)
    % The minimum g value is the most critical (closest to failure)
    state.g_min = min(g);
    state.g_critical_idx = find(g == state.g_min, 1);
    
    % Record all limit state values
    state.g_all = g;
end