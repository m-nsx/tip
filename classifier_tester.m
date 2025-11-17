close all
clc

% =====================================================================
% TEST DU MODÈLE CNN (ResNet-18 transféré)
% =====================================================================

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

fprintf('Classification en cours (%d images)...\n', numel(imdsTest.Files));

predictedLabels = classify(netTransfer, augTest);
trueLabels = imdsTest.Labels;

% =====================================================================
% Création du tableau de résultats (clé = fichier, valeur = catégorie)
% =====================================================================

results = struct('filename', {}, 'predicted', {});

for i = 1:numel(imdsTest.Files)
    [~, name, ext] = fileparts(imdsTest.Files{i});
    fileKey = strcat(name, ext); % ex: "0.jpg"
    predicted = char(predictedLabels(i));

    results(i).filename = fileKey;
    results(i).predicted = predicted;
end

% =====================================================================
% Export au format JSON
% =====================================================================

jsonStr = jsonencode(results, PrettyPrint=true);
fid = fopen('test_results.json', 'w');
fwrite(fid, jsonStr, 'char');
fclose(fid);

fprintf('Résultats sauvegardés dans "test_results.json"\n');

% =====================================================================
% Fonction d’affichage aléatoire de 16 images
% =====================================================================

function afficherEchantillon(imdsTest, predictedLabels)
    % Sélection aléatoire de 16 images
    idx = randperm(numel(imdsTest.Files), 16);

    figure('Name','Échantillon des prédictions','NumberTitle','off')
    tiledlayout(4,4, 'Padding','compact', 'TileSpacing','compact')

    for i = 1:16
        nexttile
        img = readimage(imdsTest, idx(i));
        imshow(img)
        title(string(predictedLabels(idx(i))), 'Interpreter','none', 'FontSize', 10)
    end
end

% =====================================================================
% Interaction utilisateur : afficher un échantillon quand on appuie sur une touche
% =====================================================================

disp('Appuyer sur une touche pour afficher un échantillon aléatoire (Ctrl+C pour quitter)')

while true
    pause
    afficherEchantillon(imdsTest, predictedLabels)
end