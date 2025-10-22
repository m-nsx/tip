%% =====================================================================
% Entraînement DenseNet201
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
% Chargement DenseNet201 pré-entraîné
% =====================================================================
net = nasnetlarge;
inputSize = net.Layers(1).InputSize;  % [224 224 3]
layers = net.Layers;  % <-- AJOUTEZ CETTE LIGNE

% Afficher les 3 dernières couches
disp('Les 3 dernières couches de NASNet-Large :')
for i = length(layers)-2:length(layers)
    fprintf('\nCouche %d: %s\n', i, layers(i).Name);
    disp(layers(i));
end

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

lgraph = removeLayers(lgraph, {'predictions', 'predictions_softmax', 'ClassificationLayer_predictions'});

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

% Connexion du dernier bloc DenseNet201 au nouveau fullyConnectedLayer
lgraph = connectLayers(lgraph,'global_average_pooling2d_2','fc_mid');

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
save('trainedFoodNet.mat','netTransfer');
disp('Entraînement terminé');