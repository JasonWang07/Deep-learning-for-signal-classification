function lgraph = modelResNet(modulationTypes,sps,spf)

numModTypes = numel(modulationTypes);
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];

lgraph = layerGraph();
%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    imageInputLayer([1 1024 2],"Name","imageinput")
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_49_relu")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","fc1000_softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");

plot(lgraph);




% tempLayers = [
%   imageInputLayer([1 spf 2], 'Normalization', 'none', 'Name', 'Input Layer')
%   convolution2dLayer(filterSize, 16*netWidth, 'Padding', 'same', 'Name', 'Res1')
%   batchNormalizationLayer('Name', 'BN1')
%   reluLayer('Name', 'ReLU1')
%   maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool1')];
% lgraph = addLayers(lgraph, tempLayers);
%   
% 
% tempLayers = [
%   convolution2dLayer(filterSize, 32*netWidth, 'Padding', 'same', 'Name', 'Res2L')
%   batchNormalizationLayer('Name', 'BN2L')];
% lgraph = addLayers(lgraph, tempLayers);
% 
% 
% tempLayers = [
%   convolution2dLayer(filterSize, 24*netWidth, 'Padding', 'same', 'Name', 'Res2R-1')
%   batchNormalizationLayer('Name', 'BN2R-1')
%   reluLayer('Name', 'ReLU2R-1')
%   convolution2dLayer(filterSize, 24*netWidth, 'Padding', 'same', 'Name', 'Res2R-2')
%   batchNormalizationLayer('Name', 'BN2R-2')
%   reluLayer('Name', 'ReLU2R-2')
%   convolution2dLayer(filterSize, 32*netWidth, 'Padding', 'same', 'Name', 'Res2R-3')
%   batchNormalizationLayer('Name', 'BN2R-3')];
% lgraph = addLayers(lgraph, tempLayers);
% 
% tempLayers = [
%     additionLayer(2,"Name","add_2")
%     reluLayer("Name","ReLU-2end")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% 
% tempLayers = [
%   convolution2dLayer(filterSize, 32*netWidth, 'Padding', 'same', 'Name', 'Res3R-1')
%   batchNormalizationLayer('Name', 'BN3R-1')
%   reluLayer('Name', 'ReLU3R-1')
%   convolution2dLayer(filterSize, 32*netWidth, 'Padding', 'same', 'Name', 'Res3R-2')
%   batchNormalizationLayer('Name', 'BN3R-2')
%   reluLayer('Name', 'ReLU3R-2')
%   convolution2dLayer(filterSize, 48*netWidth, 'Padding', 'same', 'Name', 'Res3R-3')
%   batchNormalizationLayer('Name', 'BN3R-3')];
% lgraph = addLayers(lgraph, tempLayers);
% 
% tempLayers = [
%     additionLayer(2,"Name","add_3")
%     reluLayer("Name","ReLU-2end")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%   convolution2dLayer(filterSize, 48*netWidth, 'Padding', 'same', 'Name', 'Res4R-1')
%   batchNormalizationLayer('Name', 'BN4R-1')
%   reluLayer('Name', 'ReLU4R-1')
%   convolution2dLayer(filterSize, 48*netWidth, 'Padding', 'same', 'Name', 'Res4R-2')
%   batchNormalizationLayer('Name', 'BN4R-2')
%   reluLayer('Name', 'ReLU4R-2')
%   convolution2dLayer(filterSize, 64*netWidth, 'Padding', 'same', 'Name', 'Res4R-3')
%   batchNormalizationLayer('Name', 'BN4R-3')];
% lgraph = addLayers(lgraph, tempLayers);
%  
% tempLayers = [
%     additionLayer(2,"Name","add_4")
%     reluLayer("Name","ReLU-3end")];
% lgraph = addLayers(lgraph,tempLayers);
% 
% tempLayers = [
%   convolution2dLayer(filterSize, 96*netWidth, 'Padding', 'same', 'Name', 'Res5L')
%   batchNormalizationLayer('Name', 'BN5L')];
% lgraph = addLayers(lgraph, tempLayers);
% 
% 
% tempLayers = [
%   convolution2dLayer(filterSize, 64*netWidth, 'Padding', 'same', 'Name', 'Res5R-1')
%   batchNormalizationLayer('Name', 'BN5R-1')
%   reluLayer('Name', 'ReLU5R-1')
%   convolution2dLayer(filterSize, 64*netWidth, 'Padding', 'same', 'Name', 'Res5R-2')
%   batchNormalizationLayer('Name', 'BN5R-2')
%   reluLayer('Name', 'ReLU5R-2')
%   convolution2dLayer(filterSize, 96*netWidth, 'Padding', 'same', 'Name', 'Res5R-3')
%   batchNormalizationLayer('Name', 'BN5R-3')];
% lgraph = addLayers(lgraph, tempLayers);
% 
% 
% tempLayers = [
%       additionLayer(2,"Name","add_final")
%       reluLayer("Name","RELU_final")
%       averagePooling2dLayer([1 ceil(spf/32)], 'Name', 'AP1')
%       fullyConnectedLayer(numModTypes, 'Name', 'FC1')
%       softmaxLayer("Name","softmax")
%       classificationLayer("Name","Output")];
% lgraph = addLayers(lgraph, tempLayers);
% 
% lgraph = connectLayers(lgraph,"MaxPool1","Res2L");
% lgraph = connectLayers(lgraph,"MaxPool1","Res2R-1");
% 
% lgraph = connectLayers(lgraph,"BN2L","add_2/in1");
% lgraph = connectLayers(lgraph,"BN2R-3","add_2/in2");
% 
% lgraph = connectLayers(lgraph,"ReLU-2end","add_3/in1");
% lgraph = connectLayers(lgraph,"ReLU-2end","Res3R-1");
% lgraph = connectLayers(lgraph,"BN3R-3","add_3/in2");
% 
% lgraph = connectLayers(lgraph,"ReLU-3end","add_4/in1");
% lgraph = connectLayers(lgraph,"ReLU-3end","Res4R-1");
% lgraph = connectLayers(lgraph,"BN4R-3","add_4/in2");
% 
% lgraph = connectLayers(lgraph,"ReLU-4end","Res5L");
% lgraph = connectLayers(lgraph,"Res5L","add_final/in1");
% lgraph = connectLayers(lgraph,"ReLU-4end","Res5R-1");
% lgraph = connectLayers(lgraph,"Res5R-3","add_final/in2");
% 
% plot(lgraph)
  
  
  
