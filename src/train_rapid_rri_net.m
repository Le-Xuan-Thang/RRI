%% Train Rapid-RRI-Net
% This script trains a deep learning model for rapid RRI prediction
% with the architecture: 1D-CNN → LSTM → ResNet → Dense → Output
% (Step 6 of the RRI workflow)

% ----------------------------------------------------------------------- 
% Author: Le Xuan Thang 
% Email: [1] lexuanthang.official@gmail.com 
%       [2] lexuanthang.official@outlook.com 
% Website: lexuanthang.vn | ORCID: 0000-0002-9911-3544 
% © 2025 by Le Xuan Thang. All rights reserved. 
% ----------------------------------------------------------------------- %

function [model, history] = train_rapid_rri_net(dataset_dir, options)
    % Train the Rapid-RRI-Net deep learning model
    %
    % Inputs:
    %   dataset_dir: Directory containing the prepared dataset (default: 'ml_dataset')
    %   options: Training options and hyperparameters
    %
    % Outputs:
    %   model: Trained model
    %   history: Training history
    
    % Check if Deep Learning Toolbox is available
    hasDL = license('test', 'Deep_Learning_Toolbox');
    
    % Set default options
    if nargin < 1
        dataset_dir = 'ml_dataset';
    end
    if nargin < 2
        options = struct();
    end
    
    % Default training options
    if ~isfield(options, 'epochs')
        options.epochs = 100;
    end
    if ~isfield(options, 'batch_size')
        options.batch_size = 32;
    end
    if ~isfield(options, 'learning_rate')
        options.learning_rate = 0.001;
    end
    if ~isfield(options, 'patience')
        options.patience = 10;
    end
    if ~isfield(options, 'validation_split')
        options.validation_split = 0.2;
    end
    if ~isfield(options, 'use_python')
        % Default to Python if Deep Learning Toolbox not available
        options.use_python = ~hasDL;
    end
    if ~isfield(options, 'model_save_path')
        options.model_save_path = 'Models/rapid_rri_net.mat';
    end
    if ~isfield(options, 'verbose')
        options.verbose = true;
    end
    
    % Create Models directory if it doesn't exist
    [model_dir, ~, ~] = fileparts(options.model_save_path);
    if ~isfolder(model_dir)
        mkdir(model_dir);
    end
    
    try
        % Load dataset
        if options.verbose
            fprintf('Loading dataset from %s...\n', dataset_dir);
        end
        
        % Load training data
        X_train = dlmread(fullfile(dataset_dir, 'train_features.csv'));
        y_train = dlmread(fullfile(dataset_dir, 'train_targets.csv'));
        
        % Load validation data if available, otherwise use validation_split
        val_file = fullfile(dataset_dir, 'val_features.csv');
        if exist(val_file, 'file')
            X_val = dlmread(val_file);
            y_val = dlmread(fullfile(dataset_dir, 'val_targets.csv'));
            has_val_set = true;
        else
            has_val_set = false;
        end
        
        % Determine input shape
        input_shape = size(X_train, 2);
        
        if options.use_python
            % Train model using Python (TensorFlow/Keras)
            [model, history] = train_with_python(X_train, y_train, X_val, y_val, has_val_set, input_shape, options);
        else
            % Train model using MATLAB's Deep Learning Toolbox
            [model, history] = train_with_matlab(X_train, y_train, X_val, y_val, has_val_set, input_shape, options);
        end
        
        % Save model
        if options.verbose
            fprintf('Saving model to %s...\n', options.model_save_path);
        end
        
        % Ensure directory exists
        model_dir = fileparts(options.model_save_path);
        if ~isfolder(model_dir)
            mkdir(model_dir);
        end
        
        % Save the model
        save(options.model_save_path, 'model', 'history', 'options', '-v7.3');
        
        % Display training summary
        display_training_summary(history, options);
        
    catch ME
        fprintf('Error training Rapid-RRI-Net: %s\n', ME.message);
        rethrow(ME);
    end
end

% Helper function: Train model using MATLAB's Deep Learning Toolbox
function [model, history] = train_with_matlab(X_train, y_train, X_val, y_val, has_val_set, input_shape, options)
    % Check if Deep Learning Toolbox is available
    if ~license('test', 'Deep_Learning_Toolbox')
        error('Deep Learning Toolbox is required for MATLAB training. Set options.use_python=true to use Python instead.');
    end
    
    if options.verbose
        fprintf('Creating Rapid-RRI-Net model with MATLAB...\n');
        fprintf('Architecture: 1D-CNN → LSTM → ResNet → Dense → Output\n');
    end
    
    % Reshape input for CNN layers (add channel dimension)
    % Convert to format: [features, sequence, samples]
    X_train_reshaped = permute(reshape(X_train', input_shape, 1, []), [1, 2, 3]);
    if has_val_set
        X_val_reshaped = permute(reshape(X_val', input_shape, 1, []), [1, 2, 3]);
    end
    
    % Create Network Architecture
    layers = [
        sequenceInputLayer(input_shape, 'Name', 'input')
        
        % 1D-CNN layers
        convolution1dLayer(5, 64, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        convolution1dLayer(5, 128, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % LSTM layer
        lstmLayer(100, 'Name', 'lstm1')
        
        % ResNet-style blocks
        additionLayer(2, 'Name', 'add1')
        
        % First branch: identity
        convolution1dLayer(1, 128, 'Padding', 'same', 'Name', 'res1_branch1')
        
        % Second branch: conv -> bn -> relu -> conv -> bn
        convolution1dLayer(3, 128, 'Padding', 'same', 'Name', 'res1_branch2a')
        batchNormalizationLayer('Name', 'res1_branch2a_bn')
        reluLayer('Name', 'res1_branch2a_relu')
        convolution1dLayer(3, 128, 'Padding', 'same', 'Name', 'res1_branch2b')
        batchNormalizationLayer('Name', 'res1_branch2b_bn')
        
        % After addition
        reluLayer('Name', 'res1_relu')
        
        % Global Pooling
        globalAveragePooling1dLayer('Name', 'gap')
        
        % Dense layers
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu_fc1')
        dropoutLayer(0.5, 'Name', 'drop1')
        fullyConnectedLayer(32, 'Name', 'fc2')
        reluLayer('Name', 'relu_fc2')
        fullyConnectedLayer(1, 'Name', 'output')
        regressionLayer('Name', 'regression')
    ];
    
    % Create layer graph for ResNet connections
    lgraph = layerGraph(layers);
    
    % Connect the ResNet skip connection
    lgraph = connectLayers(lgraph, 'lstm1', 'add1/in1');
    lgraph = connectLayers(lgraph, 'res1_branch2b_bn', 'add1/in2');
    
    % Define training options
    train_options = trainingOptions('adam', ...
        'MaxEpochs', options.epochs, ...
        'MiniBatchSize', options.batch_size, ...
        'InitialLearnRate', options.learning_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 20, ...
        'GradientThreshold', 1, ...
        'Plots', 'training-progress', ...
        'Verbose', options.verbose, ...
        'VerboseFrequency', 50);
    
    % Add validation if available
    if has_val_set
        train_options.ValidationData = {X_val_reshaped, y_val};
        train_options.ValidationFrequency = 30;
        train_options.ValidationPatience = options.patience;
    end
    
    % Train the network
    if options.verbose
        fprintf('Training Rapid-RRI-Net model with MATLAB...\n');
    end
    [model, info] = trainNetwork(X_train_reshaped, y_train, lgraph, train_options);
    
    % Extract training history
    history = struct();
    history.loss = info.TrainingLoss;
    history.epoch = 1:numel(info.TrainingLoss);
    
    if has_val_set
        history.val_loss = info.ValidationLoss;
        
        % Match validation loss length to training loss
        val_idx = round(linspace(1, numel(info.TrainingLoss), numel(info.ValidationLoss)));
        val_loss_full = zeros(numel(info.TrainingLoss), 1);
        val_loss_full(val_idx) = info.ValidationLoss;
        % Interpolate the missing validation loss values
        for i = 2:numel(val_loss_full)
            if val_loss_full(i) == 0
                % Find next non-zero value
                next_idx = find(val_loss_full(i:end) > 0, 1) + i - 1;
                if isempty(next_idx)
                    val_loss_full(i) = val_loss_full(i-1);
                else
                    % Linear interpolation
                    prev_idx = i-1;
                    val_loss_full(i) = val_loss_full(prev_idx) + ...
                        (val_loss_full(next_idx) - val_loss_full(prev_idx)) * ...
                        ((i - prev_idx) / (next_idx - prev_idx));
                end
            end
        end
        history.val_loss = val_loss_full;
    end
end

% Helper function: Train model using Python (TensorFlow/Keras)
function [model, history] = train_with_python(X_train, y_train, X_val, y_val, has_val_set, input_shape, options)
    if options.verbose
        fprintf('Creating and training Rapid-RRI-Net model with Python (TensorFlow/Keras)...\n');
    end
    
    % Create temp directory for data exchange
    temp_dir = tempdir;
    
    % Save data to temp files
    train_x_file = fullfile(temp_dir, 'train_x.csv');
    train_y_file = fullfile(temp_dir, 'train_y.csv');
    dlmwrite(train_x_file, X_train, 'precision', '%.8f');
    dlmwrite(train_y_file, y_train, 'precision', '%.8f');
    
    if has_val_set
        val_x_file = fullfile(temp_dir, 'val_x.csv');
        val_y_file = fullfile(temp_dir, 'val_y.csv');
        dlmwrite(val_x_file, X_val, 'precision', '%.8f');
        dlmwrite(val_y_file, y_val, 'precision', '%.8f');
    end
    
    % Create Python script
    py_script = fullfile(temp_dir, 'train_rapid_rri_net.py');
    model_save_path = fullfile(temp_dir, 'rapid_rri_net_model.h5');
    history_save_path = fullfile(temp_dir, 'training_history.csv');
    
    fid = fopen(py_script, 'w');
    
    % Write imports
    fprintf(fid, 'import numpy as np\n');
    fprintf(fid, 'import pandas as pd\n');
    fprintf(fid, 'import tensorflow as tf\n');
    fprintf(fid, 'import os\n');
    fprintf(fid, 'from tensorflow.keras.models import Model, Sequential\n');
    fprintf(fid, 'from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense\n');
    fprintf(fid, 'from tensorflow.keras.layers import BatchNormalization, Activation, Add, GlobalAveragePooling1D\n');
    fprintf(fid, 'from tensorflow.keras.layers import Dropout, Reshape, Layer\n');
    fprintf(fid, 'from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau\n');
    fprintf(fid, 'from tensorflow.keras.optimizers import Adam\n');
    fprintf(fid, 'import matplotlib.pyplot as plt\n\n');
    
    % Load data
    fprintf(fid, '# Load training data\n');
    fprintf(fid, 'X_train = np.loadtxt("%s", delimiter=",")\n', strrep(train_x_file, '\', '\\'));
    fprintf(fid, 'y_train = np.loadtxt("%s", delimiter=",")\n\n', strrep(train_y_file, '\', '\\'));
    
    if has_val_set
        fprintf(fid, '# Load validation data\n');
        fprintf(fid, 'X_val = np.loadtxt("%s", delimiter=",")\n', strrep(val_x_file, '\', '\\'));
        fprintf(fid, 'y_val = np.loadtxt("%s", delimiter=",")\n\n', strrep(val_y_file, '\', '\\'));
    else
        fprintf(fid, '# No separate validation set, will use validation_split\n');
        fprintf(fid, 'X_val = None\n');
        fprintf(fid, 'y_val = None\n\n');
    end
    
    % Reshape for Conv1D layers
    fprintf(fid, '# Reshape for Conv1D layers (samples, timesteps, features)\n');
    fprintf(fid, 'input_shape = %d\n', input_shape);
    fprintf(fid, 'X_train = X_train.reshape(-1, input_shape, 1)\n');
    if has_val_set
        fprintf(fid, 'X_val = X_val.reshape(-1, input_shape, 1)\n\n');
    end
    
    % Define ResNet block function
    fprintf(fid, '# Define ResNet block function\n');
    fprintf(fid, 'def residual_block(x, filters, kernel_size=3):\n');
    fprintf(fid, '    shortcut = x\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # First convolution\n');
    fprintf(fid, '    x = Conv1D(filters, kernel_size, padding="same")(x)\n');
    fprintf(fid, '    x = BatchNormalization()(x)\n');
    fprintf(fid, '    x = Activation("relu")(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Second convolution\n');
    fprintf(fid, '    x = Conv1D(filters, kernel_size, padding="same")(x)\n');
    fprintf(fid, '    x = BatchNormalization()(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Shortcut connection (if shapes don\'t match)\n');
    fprintf(fid, '    if shortcut.shape[-1] != filters:\n');
    fprintf(fid, '        shortcut = Conv1D(filters, 1, padding="same")(shortcut)\n');
    fprintf(fid, '        shortcut = BatchNormalization()(shortcut)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Add shortcut to main path\n');
    fprintf(fid, '    x = Add()([x, shortcut])\n');
    fprintf(fid, '    x = Activation("relu")(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    return x\n\n');
    
    % Build the Rapid-RRI-Net model
    fprintf(fid, '# Build the Rapid-RRI-Net model\n');
    fprintf(fid, 'def build_rapid_rri_net(input_shape):\n');
    fprintf(fid, '    # Input layer\n');
    fprintf(fid, '    inputs = Input(shape=(input_shape, 1))\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # 1D CNN layers\n');
    fprintf(fid, '    x = Conv1D(64, 5, padding="same")(inputs)\n');
    fprintf(fid, '    x = BatchNormalization()(x)\n');
    fprintf(fid, '    x = Activation("relu")(x)\n');
    fprintf(fid, '    x = MaxPooling1D(pool_size=2, strides=2)(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    x = Conv1D(128, 5, padding="same")(x)\n');
    fprintf(fid, '    x = BatchNormalization()(x)\n');
    fprintf(fid, '    x = Activation("relu")(x)\n');
    fprintf(fid, '    x = MaxPooling1D(pool_size=2, strides=2)(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # LSTM layer\n');
    fprintf(fid, '    x = LSTM(100, return_sequences=True)(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # ResNet blocks\n');
    fprintf(fid, '    x = residual_block(x, 128)\n');
    fprintf(fid, '    x = residual_block(x, 128)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Global Pooling\n');
    fprintf(fid, '    x = GlobalAveragePooling1D()(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Dense layers\n');
    fprintf(fid, '    x = Dense(64)(x)\n');
    fprintf(fid, '    x = Activation("relu")(x)\n');
    fprintf(fid, '    x = Dropout(0.5)(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    x = Dense(32)(x)\n');
    fprintf(fid, '    x = Activation("relu")(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Output layer\n');
    fprintf(fid, '    outputs = Dense(1, activation="linear")(x)\n');
    fprintf(fid, '    \n');
    fprintf(fid, '    # Create model\n');
    fprintf(fid, '    model = Model(inputs=inputs, outputs=outputs, name="Rapid-RRI-Net")\n');
    fprintf(fid, '    return model\n\n');
    
    % Create the model
    fprintf(fid, '# Create the model\n');
    fprintf(fid, 'model = build_rapid_rri_net(input_shape)\n');
    fprintf(fid, 'model.summary()\n\n');
    
    % Compile the model
    fprintf(fid, '# Compile the model\n');
    fprintf(fid, 'model.compile(\n');
    fprintf(fid, '    optimizer=Adam(learning_rate=%f),\n', options.learning_rate);
    fprintf(fid, '    loss="mse",\n');
    fprintf(fid, '    metrics=["mae"]\n');
    fprintf(fid, ')\n\n');
    
    % Define callbacks
    fprintf(fid, '# Define callbacks\n');
    fprintf(fid, 'callbacks = [\n');
    fprintf(fid, '    EarlyStopping(monitor="val_loss", patience=%d, restore_best_weights=True),\n', options.patience);
    fprintf(fid, '    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=%d, min_lr=1e-6),\n', options.patience/2);
    fprintf(fid, '    ModelCheckpoint("%s", monitor="val_loss", save_best_only=True, verbose=1)\n', strrep(model_save_path, '\', '\\'));
    fprintf(fid, ']\n\n');
    
    % Train the model
    fprintf(fid, '# Train the model\n');
    if has_val_set
        fprintf(fid, 'history = model.fit(\n');
        fprintf(fid, '    X_train, y_train,\n');
        fprintf(fid, '    validation_data=(X_val, y_val),\n');
        fprintf(fid, '    epochs=%d,\n', options.epochs);
        fprintf(fid, '    batch_size=%d,\n', options.batch_size);
        fprintf(fid, '    callbacks=callbacks,\n');
        fprintf(fid, '    verbose=1\n');
        fprintf(fid, ')\n\n');
    else
        fprintf(fid, 'history = model.fit(\n');
        fprintf(fid, '    X_train, y_train,\n');
        fprintf(fid, '    validation_split=%f,\n', options.validation_split);
        fprintf(fid, '    epochs=%d,\n', options.epochs);
        fprintf(fid, '    batch_size=%d,\n', options.batch_size);
        fprintf(fid, '    callbacks=callbacks,\n');
        fprintf(fid, '    verbose=1\n');
        fprintf(fid, ')\n\n');
    end
    
    % Save training history
    fprintf(fid, '# Save training history\n');
    fprintf(fid, 'history_df = pd.DataFrame(history.history)\n');
    fprintf(fid, 'history_df["epoch"] = history.epoch\n');
    fprintf(fid, 'history_df.to_csv("%s", index=False)\n\n', strrep(history_save_path, '\', '\\'));
    
    % Create learning curve plot
    fprintf(fid, '# Create learning curve plot\n');
    fprintf(fid, 'plt.figure(figsize=(12, 4))\n');
    fprintf(fid, 'plt.subplot(1, 2, 1)\n');
    fprintf(fid, 'plt.plot(history.history["loss"], label="Training Loss")\n');
    fprintf(fid, 'plt.plot(history.history["val_loss"], label="Validation Loss")\n');
    fprintf(fid, 'plt.xlabel("Epoch")\n');
    fprintf(fid, 'plt.ylabel("Loss")\n');
    fprintf(fid, 'plt.legend()\n');
    fprintf(fid, 'plt.title("Loss")\n');
    fprintf(fid, 'plt.grid(True)\n\n');
    
    fprintf(fid, 'plt.subplot(1, 2, 2)\n');
    fprintf(fid, 'plt.plot(history.history["mae"], label="Training MAE")\n');
    fprintf(fid, 'plt.plot(history.history["val_mae"], label="Validation MAE")\n');
    fprintf(fid, 'plt.xlabel("Epoch")\n');
    fprintf(fid, 'plt.ylabel("Mean Absolute Error")\n');
    fprintf(fid, 'plt.legend()\n');
    fprintf(fid, 'plt.title("MAE")\n');
    fprintf(fid, 'plt.grid(True)\n\n');
    
    fprintf(fid, 'plt.tight_layout()\n');
    fprintf(fid, 'plt.savefig("%s")\n', strrep(fullfile(temp_dir, 'learning_curves.png'), '\', '\\'));
    
    fclose(fid);
    
    % Execute Python script
    if options.verbose
        fprintf('Running Python training script...\n');
    end
    
    % Run Python script
    [status, cmdout] = system(['python "', py_script, '"']);
    
    if status ~= 0
        error('Error executing Python script: %s', cmdout);
    end
    
    % Load training history
    history_data = readtable(history_save_path);
    
    % Create history structure
    history = struct();
    history.epoch = history_data.epoch;
    history.loss = history_data.loss;
    history.val_loss = history_data.val_loss;
    history.mae = history_data.mae;
    history.val_mae = history_data.val_mae;
    
    % Create a simple model wrapper for MATLAB
    model = struct();
    model.python_model_path = model_save_path;
    
    % Define a custom predict function
    model.predict = @(X) predict_with_python(X, model_save_path);
    
    if options.verbose
        fprintf('Python training completed successfully.\n');
    end
    
    % Clean up temporary files
    if ~options.verbose  % Keep files if verbose for debugging
        delete(train_x_file);
        delete(train_y_file);
        if has_val_set
            delete(val_x_file);
            delete(val_y_file);
        end
        % Don't delete model and history files as they're needed
    end
end

% Helper function: Predict using Python model
function y_pred = predict_with_python(X, model_path)
    % Create temp directory for data exchange
    temp_dir = tempdir;
    
    % Save input data
    input_file = fullfile(temp_dir, 'predict_input.csv');
    output_file = fullfile(temp_dir, 'predict_output.csv');
    dlmwrite(input_file, X, 'precision', '%.8f');
    
    % Create Python prediction script
    py_script = fullfile(temp_dir, 'predict.py');
    
    fid = fopen(py_script, 'w');
    
    fprintf(fid, 'import numpy as np\n');
    fprintf(fid, 'import tensorflow as tf\n\n');
    
    fprintf(fid, '# Load input data\n');
    fprintf(fid, 'X = np.loadtxt("%s", delimiter=",")\n\n', strrep(input_file, '\', '\\'));
    
    fprintf(fid, '# Reshape for Conv1D\n');
    fprintf(fid, 'X = X.reshape(-1, X.shape[1], 1)\n\n');
    
    fprintf(fid, '# Load model\n');
    fprintf(fid, 'model = tf.keras.models.load_model("%s")\n\n', strrep(model_path, '\', '\\'));
    
    fprintf(fid, '# Make predictions\n');
    fprintf(fid, 'predictions = model.predict(X)\n\n');
    
    fprintf(fid, '# Save predictions\n');
    fprintf(fid, 'np.savetxt("%s", predictions, delimiter=",")\n', strrep(output_file, '\', '\\'));
    
    fclose(fid);
    
    % Execute prediction script
    [status, cmdout] = system(['python "', py_script, '"']);
    
    if status ~= 0
        error('Error making predictions with Python model: %s', cmdout);
    end
    
    % Load predictions
    y_pred = dlmread(output_file);
    
    % Clean up
    delete(input_file);
    delete(py_script);
    delete(output_file);
end

% Helper function: Display training summary
function display_training_summary(history, options)
    fprintf('\n--------------------------------------------------\n');
    fprintf('Rapid-RRI-Net Training Summary\n');
    fprintf('--------------------------------------------------\n');
    
    % Display final metrics
    fprintf('Final training loss:   %.6f\n', history.loss(end));
    
    if isfield(history, 'val_loss')
        fprintf('Final validation loss: %.6f\n', history.val_loss(end));
    end
    
    if isfield(history, 'mae')
        fprintf('Final training MAE:    %.6f\n', history.mae(end));
    end
    
    if isfield(history, 'val_mae')
        fprintf('Final validation MAE:  %.6f\n', history.val_mae(end));
    end
    
    % Display training settings
    fprintf('\nTraining settings:\n');
    fprintf('Epochs:               %d\n', options.epochs);
    fprintf('Batch size:           %d\n', options.batch_size);
    fprintf('Learning rate:        %.6f\n', options.learning_rate);
    fprintf('Patience:             %d\n', options.patience);
    
    % Determine if early stopping occurred
    if length(history.epoch) < options.epochs
        fprintf('Early stopping:       Yes (stopped at epoch %d)\n', length(history.epoch));
    else
        fprintf('Early stopping:       No (completed all %d epochs)\n', options.epochs);
    end
    
    fprintf('--------------------------------------------------\n');
    
    % Plot training curves
    figure('Position', [100, 100, 1000, 400]);
    
    % Loss curve
    subplot(1, 2, 1);
    plot(history.epoch, history.loss, 'b-', 'LineWidth', 2);
    hold on;
    if isfield(history, 'val_loss')
        plot(history.epoch, history.val_loss, 'r--', 'LineWidth', 2);
        legend('Training', 'Validation');
    end
    xlabel('Epoch');
    ylabel('Loss (MSE)');
    title('Training and Validation Loss');
    grid on;
    
    % MAE curve
    subplot(1, 2, 2);
    if isfield(history, 'mae')
        plot(history.epoch, history.mae, 'b-', 'LineWidth', 2);
        hold on;
    end
    if isfield(history, 'val_mae')
        plot(history.epoch, history.val_mae, 'r--', 'LineWidth', 2);
        legend('Training', 'Validation');
    end
    xlabel('Epoch');
    ylabel('Mean Absolute Error');
    title('Training and Validation MAE');
    grid on;
    
    % Save figure
    saveas(gcf, 'Figures/training_curves.svg');
    fprintf('Training curves saved to Figures/training_curves.svg\n');
end