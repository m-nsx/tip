%% =====================================================================
% Entraînement EfficientNet-B0
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

%% =====================================================================
% Chargement EfficientNet-B0 pré-entraîné
% =====================================================================
net = efficientnetb0;
inputSize = net.Layers(1).InputSize;  % [224 224 3]

%% =====================================================================
% Data augmentation géométrique
% =====================================================================
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20 20], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXScale', [0.85 1.15], ...
    'RandYScale', [0.85 1.15], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'OutputSizeMode', 'resize', ...
    'DispatchInBackground', true);

augValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'OutputSizeMode', 'resize');

%% =====================================================================
% Personnalisation des couches finales
% =====================================================================
lgraph = layerGraph(net);
numClasses = numel(categories(imdsTrain.Labels));

% ✅ Supprimer les couches finales d’EfficientNet-B0
% (Les noms des dernières couches peuvent varier selon la version MATLAB)
lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|dense|MatMul', ...
                               'efficientnet-b0|model|head|dense|BiasAdd', ...
                               'ClassificationLayer_predictions'});

% Nouvelles couches de classification
newLayers = [
    fullyConnectedLayer(1024,'Name','fc_mid', ...
        'WeightLearnRateFactor',5,'BiasLearnRateFactor',5)
    reluLayer('Name','relu_mid')
    dropoutLayer(0.5,'Name','dropout_food')
    fullyConnectedLayer(numClasses,'Name','fc_food', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

lgraph = addLayers(lgraph,newLayers);

% Connexion du dernier bloc EfficientNet-B0 au nouveau fullyConnectedLayer
lgraph = connectLayers(lgraph,'efficientnet-b0|model|head|global_pool','fc_mid');

%% =====================================================================
% Fine-tuning : débloquer les dernières couches
% =====================================================================
layersToUnfreeze = 10;
for i = numel(lgraph.Layers)-layersToUnfreeze:numel(lgraph.Layers)
    layer = lgraph.Layers(i);
    if isprop(layer,'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = 2;
        layer.BiasLearnRateFactor = 2;
    end
end

%% =====================================================================
% Options d'entraînement
% =====================================================================
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 3e-5, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', 100, ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4, ...
    'ValidationPatience', 5);

%% =====================================================================
% Entraînement
% =====================================================================
netTransfer = trainNetwork(augTrain, lgraph, options);

%% =====================================================================
% Sauvegarde
% =====================================================================
save('trainedFoodNet_EfficientNetB0.mat','netTransfer');
disp('Entraînement EfficientNet-B0 terminé');
