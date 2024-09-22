function modClassLSTM = modelLSTM(modulationTypes, spf)

numFeatures = [1 spf 2];
numHiddenUnits = 50;
numClasses = numel(modulationTypes);
modClassLSTM = [
    sequenceInputLayer(numFeatures)
    flattenLayer
    lstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.2
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];
end 