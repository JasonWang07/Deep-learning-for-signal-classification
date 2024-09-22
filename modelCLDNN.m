function modClassNet  = modelCLDNN(modulationTypes,sps,spf)

numHiddenUnits = 100;

%helperModClassCNN Modulation classification CNN
%   CNN = helperModClassCNN(MODTYPES,SPS,SPF) creates a convolutional
%   neural network, CNN, for modulation classification. MODTYPES is the
%   modulation types the network can identify, SPS is the samples per
%   symbol, and SPF is the samples per frame.
%
%   The CNN consists of six convolution layers and one fully connected
%   layer. Each convolution layer except the last is followed by a batch
%   normalization layer, rectified linear unit (ReLU) activation layer, and
%   max pooling layer. In the last convolution layer, the max pooling layer
%   is replaced with an average pooling layer. The output layer has softmax
%   activation. 
%   
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019 The MathWorks, Inc.

numModTypes = numel(modulationTypes);
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];
modClassNet = [
  imageInputLayer([1 spf 2], 'Normalization', 'none', 'Name', 'Input Layer')
  
  convolution2dLayer(filterSize, 16*netWidth, 'Padding', 'same', 'Name', 'CNN1')
  batchNormalizationLayer('Name', 'BN1')
  reluLayer('Name', 'ReLU1')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool1')
  
  convolution2dLayer(filterSize, 24*netWidth, 'Padding', 'same', 'Name', 'CNN2')
  batchNormalizationLayer('Name', 'BN2')
  reluLayer('Name', 'ReLU2')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool2')

  averagePooling2dLayer([1 ceil(spf/32)], 'Name', 'AP1')
  
  flattenLayer("Name","flatten")

  lstmLayer(numHiddenUnits,'OutputMode','last')
  dropoutLayer(0.4)
  
  fullyConnectedLayer(numModTypes, 'Name', 'FC1')
  softmaxLayer('Name', 'SoftMax')
  
  classificationLayer('Name', 'Output') ];


