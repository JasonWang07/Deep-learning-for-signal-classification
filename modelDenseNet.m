function lgraph = modelDenseNet(modulationTypes,sps,spf)

numModTypes = numel(modulationTypes);
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([1 1024 2],"Name","imageinput")
    convolution2dLayer([7 7],64,"Name","conv1|conv","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","conv1|bn")
    reluLayer("Name","conv1|relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block1_0_bn")
    reluLayer("Name","conv2_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block1_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block1_1_bn")
    reluLayer("Name","conv2_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block1_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block2_0_bn")
    reluLayer("Name","conv2_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block2_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block2_1_bn")
    reluLayer("Name","conv2_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block2_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block2_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block3_0_bn")
    reluLayer("Name","conv2_block3_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block3_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block3_1_bn")
    reluLayer("Name","conv2_block3_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block3_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block3_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block4_0_bn")
    reluLayer("Name","conv2_block4_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block4_1_conv","BiasLearnRateFactor",0)
    batchNormalizationLayer("Name","conv2_block4_1_bn")
    reluLayer("Name","conv2_block4_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block4_2_conv","BiasLearnRateFactor",0,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv2_block4_concat")
    batchNormalizationLayer("Name","batchnorm")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(10,"Name","fc")
    softmaxLayer("Name","fc1000_softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"pool1","conv2_block1_0_bn");
lgraph = connectLayers(lgraph,"pool1","conv2_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block1_2_conv","conv2_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block1_concat","conv2_block2_0_bn");
lgraph = connectLayers(lgraph,"conv2_block1_concat","conv2_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block2_2_conv","conv2_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block2_concat","conv2_block3_0_bn");
lgraph = connectLayers(lgraph,"conv2_block2_concat","conv2_block3_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block3_2_conv","conv2_block3_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block3_concat","conv2_block4_0_bn");
lgraph = connectLayers(lgraph,"conv2_block3_concat","conv2_block4_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block4_2_conv","conv2_block4_concat/in2");

plot(lgraph);