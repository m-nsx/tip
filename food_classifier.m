    
close all


%% =====================================================================
% Chargement des données
% =====================================================================
trainFolder = fullfile('dataset','train');
imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.85, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));

layers = [  imageInputLayer([32 32 3],'Name','input')
            convolution2dLayer(24,6,'Padding','same','Name','conv1')
            batchNormalizationLayer
            reluLayer('Name','relu1')
            maxPooling2dLayer(2, 'Stride', 2, 'Name','pool1')
            convolution2dLayer(12,16,'Padding','same','Name','conv2')
            batchNormalizationLayer
            reluLayer('Name','relu2')
            maxPooling2dLayer(2, 'Stride', 2, 'Name','pool2')

            fullyConnectedLayer(240,'Name','fc_mid', ...
                'WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
            reluLayer('Name','relu_mid')
            fullyConnectedLayer(numClasses,'Name','fc_food', ...
                'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
            softmaxLayer('Name','softmax')
            classificationLayer('Name','output')];
       
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale', [0.85 1.15], ...
    'RandYScale', [0.85 1.15], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore([32 32], imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'OutputSizeMode', 'resize', ...
    'DispatchInBackground', true);

augValidation = augmentedImageDatastore([32 32], imdsValidation, ...
    'OutputSizeMode', 'resize');





%% =====================================================================
% Options d'entraînement
% =====================================================================
miniBatchSize = 128;

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 12, ...
    'InitialLearnRate', 3e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 100, ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress', ...
    'ValidationPatience', 5);

%% =====================================================================
% Entraînement
% =====================================================================
netTransfer = trainNetwork(augTrain, layers, options);

%% =====================================================================
% Matrice de confusion sur l'ensemble de validation
% =====================================================================

% Prédictions
YPred = classify(netTransfer, augValidation);
YValidation = imdsValidation.Labels;

% Matrice de confusion
figure;
cm = confusionchart(YValidation, YPred);
cm.Title = 'Matrice de confusion - Validation';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% =====================================================================
% Sauvegarde
% =====================================================================
save('trainedFoodNet.mat','netTransfer');
disp('Entraînement terminé');