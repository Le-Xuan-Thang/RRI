% Load the training and validation data
load('training_data.mat');
load('validation_data.mat');

% Train the model
model = trainModel(training_data);

% Evaluate the model
[accuracy, mae, loss] = evaluateModel(model, validation_data);

% Store metrics
metrics.accuracy = accuracy;
metrics.mae = mae;
metrics.loss = loss;

% Plot confusion matrix
confusionMatrix = plotConfusionMatrix(model, validation_data);

% Save metrics and confusion matrix
save('metrics.mat', 'metrics');
save('confusion_matrix.mat', 'confusionMatrix');

% Plot training and validation metrics
figure;
plot(metrics.training_loss, 'r', 'DisplayName', 'Training Loss');
hold on;
plot(metrics.validation_loss, 'b', 'DisplayName', 'Validation Loss');
legend;
xlabel('Epoch');
ylabel('Loss');
title('Training and Validation Loss');

% Save the figure
saveas(gcf, 'training_validation_loss.png');