%% ------------------------------------------------------------
% Matrice de confusion pour trainedFoodNet.mat
% ------------------------------------------------------------

clc; clear; close all;

%% ------------------------------------------------------------
% Catégories officielles du dataset
% ------------------------------------------------------------
foodClasses = [
    "bread"
    "dairy"
    "rice"
    "egg"
    "meat"
    "dessert"
    "fried"
    "noodles-pasta"
    "seafood"
    "soup"
    "vegetable-fruit"
];

%% ------------------------------------------------------------
% Chargement du dataset d'entraînement
% ------------------------------------------------------------
trainDir = fullfile('dataset','train');

imdsTrain = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('Images entraînement : %d\n', numel(imdsTrain.Files));

%% ------------------------------------------------------------
% Chargement automatique du réseau
% ------------------------------------------------------------
vars = load('trainedFoodNet.mat');
varNames = fieldnames(vars);

net = vars.(varNames{1});   % on prend la 1ère variable du .mat

fprintf('Réseau chargé : %s\n', varNames{1});

%% ------------------------------------------------------------
% Détermination de la taille d'entrée
% ------------------------------------------------------------
if isprop(net, 'Layers')
    % SeriesNetwork
    inputSize = net.Layers(1).InputSize(1:2);
elseif isprop(net, 'InputSizes')
    % DAGNetwork
    inputSize = net.Layers(1).InputSize(1:2);
elseif isprop(net, 'InputSize')
    inputSize = net.InputSize(1:2);
else
    error('Impossible de déterminer la taille d''entrée du réseau.');
end

%% ------------------------------------------------------------
% Prétraitement
% ------------------------------------------------------------
augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing','none');

%% ------------------------------------------------------------
% Prédictions complètes
% ------------------------------------------------------------
YPred_raw = classify(net, augTrain);
YTrue     = imdsTrain.Labels;

%% ------------------------------------------------------------
% Filtrage des labels pour ne garder que les 8 classes Food
% ------------------------------------------------------------
mask = ismember(YPred_raw, foodClasses);

YPred = YPred_raw(mask);
YTrue = YTrue(mask);

fprintf('Échantillons valides gardés : %d\n', numel(YPred));

%% ------------------------------------------------------------
% Matrice de confusion
% ------------------------------------------------------------
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Matrice de confusion - trainedFoodNet';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

fprintf('Matrice de confusion générée.\n');
