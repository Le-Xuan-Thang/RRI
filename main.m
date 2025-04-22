%% RRI Project Main Entry Point
% This script serves as the main entry point for the Reliability-Robustness Index (RRI) project
% All code files are located in the src folder

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

%% Initialize Environment
% Clear workspace and command window
clear;
clc;
close all;

% Reset MATLAB path to default before adding new paths
restoredefaultpath;
rehash toolboxcache;

% Get current directory
current_dir = pwd;

% Add necessary paths with full paths to avoid conflicts
addpath(fullfile(current_dir, 'src'));
addpath(fullfile(current_dir, 'Generate Acc'));
% Critical addition: Add STABIL toolbox path
addpath("/Applications/MATLAB_R2024b.app/toolbox_Stabil");
fprintf('Reset MATLAB path to default and added required folders to path\n');

% Display project information
fprintf('\n======================================================\n');
fprintf('Reliability-Robustness Index (RRI) Project\n');
fprintf('======================================================\n');
fprintf('All code files are located in the src folder\n\n');

%% Run the main steps of the RRI workflow
try
    % Check if we should run the full RRI calculation first (steps 1-4)
    run_full_workflow = input('Run full RRI calculation (steps 1-4)? (y/n): ', 's');
    
    if strcmpi(run_full_workflow, 'y')
        fprintf('Running main RRI calculation (Steps 1-4 of the RRI workflow)...\n\n');
        run(fullfile(current_dir, 'main_RRI.m'));
        fprintf('RRI calculation completed. Now continuing with deep learning steps...\n\n');
    end
    
    % Run the deep learning workflow script (steps 5-7)
    fprintf('Running deep learning workflow (Steps 5-7 of the RRI workflow)...\n\n');
    
    % Run main_steps_5_to_7.m with full path to avoid conflicts
    run(fullfile(current_dir, 'main_steps_5_to_7.m'));
    
    % Display success message
    fprintf('\n======================================================\n');
    fprintf('RRI Project execution completed successfully!\n');
    fprintf('======================================================\n');
catch ME
    % Display error message
    fprintf('\n======================================================\n');
    fprintf('Error occurred during execution:\n');
    fprintf('%s\n', ME.message);
    fprintf('Error in file: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    fprintf('======================================================\n');
    
    % Rethrow the error for debugging
    rethrow(ME);
end