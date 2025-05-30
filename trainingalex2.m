inputSize = [227 227 3];

myReadFcn = @(filename) imresize(imread(filename), inputSize(1:2));

imdsAll = imageDatastore('gt_db', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', myReadFcn);

labels = unique(imdsAll.Labels);
selectedFiles = [];
selectedLabels = [];
filesPerLabel = 10;

for i = 1:numel(labels)
    idx = find(imdsAll.Labels == labels(i));
    idx = idx(1:min(filesPerLabel, numel(idx)));  
    selectedFiles = [selectedFiles; imdsAll.Files(idx)];
    selectedLabels = [selectedLabels; imdsAll.Labels(idx)];
end

imds = imageDatastore(selectedFiles, 'ReadFcn', myReadFcn);
imds.Labels = selectedLabels;

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
figure;
for i = 1:16
    subplot(4, 4, i);
    I = readimage(imdsTrain, idx(i));
    imshow(I);
    title(string(imdsTrain.Labels(idx(i))));
end

net = alexnet;
inputSize = net.Layers(1).InputSize;

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses, ...
    'Name', 'fc8', ...
    'WeightLearnRateFactor', 20, ...
    'BiasLearnRateFactor', 20);
layers(end) = classificationLayer('Name', 'output');

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

options = trainingOptions("adam", ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 3e-4, ...  
    'Shuffle', "every-epoch", ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 8, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', false, ...
    'Plots', "training-progress");




% Step 9: Train the network
trainedNet = trainNetwork(augimdsTrain, layers, options);
