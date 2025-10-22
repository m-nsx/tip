close all
clc

fprint("")

% Charger le réseau entraîné
load('trainedFoodNet.mat', 'netTransfer')

% Dossier de test
testFolder = fullfile('dataset', 'test');

% Chargement des images de test
imdsTest = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Taille d'entrée du réseau
inputSize = netTransfer.Layers(1).InputSize(1:2);

% Préparation du datastore
augTest = augmentedImageDatastore(inputSize, imdsTest);

% =====================================================================
% Classification du jeu de test
% =====================================================================

[predictedLabels, scores] = classify(netTransfer, augTest);
trueLabels = imdsTest.Labels;

% =====================================================================
% Création d'une table de résultats
% =====================================================================

results = struct('filename', {}, 'predicted', {}, 'confidence', {});

for i = 1:numel(imdsTest.Files)
    [~, name, ext] = fileparts(imdsTest.Files{i});
    fileKey = strcat(name, ext);
    [~, maxIdx] = max(scores(i,:));
    conf = scores(i,maxIdx);
    results(end+1) = struct( ...
        'filename', fileKey, ...
        'predicted', char(predictedLabels(i)), ...
        'confidence', conf ...
    );
end

% =====================================================================
% Export au format JSON
% =====================================================================

jsonStr = jsonencode(results, PrettyPrint=true);
fid = fopen('test_results.json', 'w');
fwrite(fid, jsonStr, 'char');
fclose(fid);

% =====================================================================
% Fonction d’affichage aléatoire de 16 images
% =====================================================================

function afficherEchantillon(imdsTest, predictedLabels, scores)
    idx = randperm(numel(imdsTest.Files), 16);

    figure('Name','Échantillon des prédictions','NumberTitle','off')
    tiledlayout(4,4, 'Padding','compact', 'TileSpacing','compact')

    for i = 1:16
        nexttile
        img = readimage(imdsTest, idx(i));
        imshow(img)
        [~, maxIdx] = max(scores(idx(i),:));
        conf = scores(idx(i), maxIdx) * 100;
        title(sprintf('%s (%.1f%%)', string(predictedLabels(idx(i))), conf), ...
            'Interpreter','none', 'FontSize', 9)
    end
end

% =====================================================================
% Interaction utilisateur : afficher un échantillon quand on appuie sur une touche
% =====================================================================

disp('Appuyer sur une touche')

while true
    pause
    afficherEchantillon(imdsTest, predictedLabels, scores)
end