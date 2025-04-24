function load_packages(current_dir)

    if ~exist(fullfile(current_dir, 'src'), 'dir')
        warning('Directory "src" does not exist in the current path.');
        disp('Cloning the repository...');
        system('git clone https://github.com/Le-Xuan-Thang/RRI.git');
    end
    addpath(fullfile(current_dir, 'src'));

    if ~exist(fullfile(current_dir, 'Generate Acc'), 'dir')
        warning('Directory "Generate Acc" does not exist in the current path.');
        disp('Cloning the repository...');
        system('git clone https://github.com/Le-Xuan-Thang/RRI.git');
    end
    addpath(fullfile(current_dir, 'Generate Acc'));
   
    % Define the stabil folder path
    stabil_folder = fullfile(current_dir, 'stabil');
    % Check if the OAs folder exists; if not, clone it from GitHub
    if ~isfolder(stabil_folder)
        fprintf('stabil_folder not found. Cloning repository from GitHub...\n');
        [status, result] = system('git clone https://github.com/Le-Xuan-Thang/stabil.git');
    else
        fprintf('Pull new updates from GitHub...\n');
        cd(stabil_folder);
        [status, result] = system('git pull origin main');
        fprintf('stabil folder already exists. Checking for updates...\n');
        cd(current_dir);
    end
    if status ~= 0
        error('Failed to clone stabil repository. Details: %s', result);
    else
        fprintf('Repository cloned successfully.\n');
    end
    addpath(fullfile(current_dir, 'stabil'));


    if ~exist(fullfile(current_dir, 'Data'), 'dir')
        mkdir(fullfile(current_dir, 'Data'));
    end
    % check Figures
    if ~exist(fullfile(current_dir, 'Figures'), 'dir')
        mkdir(fullfile(current_dir, 'Figures'));
    end

    disp('All required packages are loaded successfully.');
end