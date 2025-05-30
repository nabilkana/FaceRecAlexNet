s = load('trainedModel2.mat');  
net = s.trainedNet;

inputSize = net.Layers(1).InputSize;
numFeatures = net.Layers(20).OutputSize;

features = cell(50, 1);

for i = 1:50
    folderPath = fullfile('gt_db', sprintf('s%02d', i));
    imds = imageDatastore(folderPath, 'FileExtensions', {'.jpg', '.png', '.pgm'});

    featureMatrix = zeros(numFeatures, 5);  

    for j = 11:15
        img = readimage(imds, j);
        if size(img, 3) == 1
            img = repmat(img, [1, 1, 3]);  
        end
        img = imresize(img, inputSize(1:2));
        
        layerOutputs = activations(net, img, 'fc7', 'OutputAs', 'columns');
        featureMatrix(:, j - 10) = layerOutputs;
    end
    features{i} = featureMatrix;
end

genuine_scores = [];
imposter_scores = [];

for i = 1:50
    F = features{i};
    refFeature = F(:, 1);  

    for j = 2:5
        score = cosine_similarity(refFeature, F(:, j));
        genuine_scores = [genuine_scores; score];
    end

    for j = 1:50
        if j == i, continue; end
        F_other = features{j};
        for k = 2:5
            score = cosine_similarity(refFeature, F_other(:, k));
            imposter_scores = [imposter_scores; score];
        end
    end
end

labels = [ones(size(genuine_scores)); zeros(size(imposter_scores))];
all_scores = [genuine_scores; imposter_scores];

[X, Y, T, AUC] = perfcurve(labels, all_scores, 1);
figure;
plot(X, Y, 'LineWidth', 1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC, '%.4f') ')']);
grid on;

figure;
histogram(genuine_scores, 'BinWidth', 0.02, 'Normalization', 'probability', 'FaceColor', 'g');
hold on;
histogram(imposter_scores, 'BinWidth', 0.02, 'Normalization', 'probability', 'FaceColor', 'r');
legend('Genuine', 'Imposter');
xlabel('Cosine Similarity Score');
ylabel('Probability');
title('Genuine vs Imposter Score Distribution');
grid on;

function sim = cosine_similarity(vec1, vec2)
    sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2));
end
mean_genuine = mean(genuine_scores);
mean_imposter = mean(imposter_scores);
std_genuine = std(genuine_scores);
std_imposter = std(imposter_scores);

d_prime = (mean_genuine - mean_imposter) / sqrt(0.5 * (std_genuine^2 + std_imposter^2));

fprintf('d-prime (d′) = %.4f\n', d_prime);
zeroFPR_idx = find(X == 0);

TPR_at_zeroFPR = Y(zeroFPR_idx);

targetTPR = 0.955;
valid_idx = find(TPR_at_zeroFPR <= targetTPR, 1, 'last');

bestThreshold = T(zeroFPR_idx(valid_idx));
