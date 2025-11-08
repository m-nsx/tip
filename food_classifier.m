%% =====================================================================
% Réseau léger "ReLU-Net" entraîné depuis zéro
% =====================================================================

close all
clc

%% =====================================================================
% Chargement des données
% =====================================================================
trainFolder = fullfile('dataset','train');
imds = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.85, 'randomized');
numClasses = numel(categories(imdsTrain.Labels));

%% =====================================================================
% Paramètres de base
% =====================================================================
inputSize = [128 128 3];  % 🔹 plus petit que 224 pour aller plus vite

%% =====================================================================
% Réseau convolutionnel léger avec ReLU
% =====================================================================
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zscore')
    
    % --- Bloc 1 ---
    convolution2dLayer(3, 16, 'Padding', 'same', 'Stride', 1, 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % --- Bloc 2 ---
    convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 1, 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % --- Bloc 3 ---
    convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', 1, 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    % --- Bloc fully connected ---
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'dropout')
    
    % --- Sortie ---
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% =====================================================================
% Data augmentation légère
% =====================================================================
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% =====================================================================
% Options d'entraînement
% =====================================================================
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 25, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 100, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% =====================================================================
% Entraînement du modèle
% =====================================================================
netLightReLU = trainNetwork(augTrain, layers, options);

%% =====================================================================
% Sauvegarde
% =====================================================================
save('ReLU_Net_Light.mat', 'netLightReLU');
disp('✅ Entraînement du réseau léger ReLU terminé');
