clear all; clc;

%%
[Y0,C,Ytest,Ctest] = getCFAR10(50);
Y0 = Y0';
nImg = [32 32];
cin = 3;


%%
% choose file for results and specify whether or not to retrain
resFile = sprintf('%s.mat',mfilename);
doTrain = true;

% set GPU flag and precision
useGPU = 0;
precision='double';

[Y0,C,Ytest,Ctest] = gpuVar(useGPU,precision,Y0',C,Ytest,Ctest);

%% choose convolution
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv    = @convFFT;
end

%% setup network
doNorm = 1;
% nLayer = @getTVNormLayer
nLayer = @getBatchNormLayer;

miniBatchSize = 50;
act1 = @tanhActivation;
actc = @tanhActivation;
act  = @tanhActivation;

nf  = [16;32;64;64];
nt  = 2*[1;1;1];
h  = 1*[1;1;1];

blocks    = cell(0,1); RegOps = cell(0,1);

%% Block to open the network
nL = nLayer([prod(nImg(1:2)) nf(1) miniBatchSize],'isWeight',1);
blocks{end+1} = NN({singleLayer(conv(nImg,[3 3 cin nf(1)]),'activation', act1,'nLayer',nL)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
%% UNIT
for k=1:numel(h)
    nImgc = nImg/(2^(k-1));
    % implicit layer
    K = conv(nImgc,[3 3 nf(k) nf(k)]);
    
    nL = nLayer([prod(nImgc(1:2)) nf(k) miniBatchSize],'isWeight',1);
    layer = doubleSymLayer(K,'activation',act,'nLayer',nL);
    blocks{end+1} = ResNN(layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
    regD = gpuVar(useGPU,precision,repmat([ones(nTheta(K),1); zeros(nTheta(nL),1)],nt(k),1));
    RegOps{end+1} = opDiag(regD);
    % Connector block
        
    nL = nLayer([prod(nImgc(1:2)) nf(k+1) miniBatchSize], 'isWeight',1);
    Kc = conv(nImgc,[1,1,nf(k),nf(k+1)]);
    blocks{end+1} = NN({singleLayer(Kc,'activation',actc,'nLayer',nL)},'useGPU',useGPU,'precision',precision);
    regD = gpuVar(useGPU,precision,[ones(nTheta(Kc),1); zeros(nTheta(nL),1)]);
    RegOps{end+1} = opDiag(regD);
    
    if k<numel(h)
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],2));
        RegOps{end+1} = opEye(nTheta(blocks{end}));
    end
end

%% Connector block
B = gpuVar(useGPU,precision,kron(eye(nf(k+1)),ones(prod(nImgc),1)));
blocks{end+1} = connector(B'/prod(nImgc));
RegOps{end+1} = opEye(nTheta(blocks{end}));

%% Put it all together
net   = MegaNet(blocks);
pLoss = softmaxLoss();

if true
theta  = initTheta(net);
W      = 0.1*vec(randn(10,nFeatOut(net)+1));
W = min(W,.2);
W = max(W,-.2);
else
    fprintf('load TF data\n');
    [theta,W] = loadTFweights('stl_finalweights.mat',2,2);
    [theta,W] = gpuVar(useGPU,precision,theta,W);
end

%%
tic; 
for k=1:10
    Z = apply(net,theta,Y0);
end
timeMAT = toc/10;

save('/Users/lruthot/Dropbox/DynamicCNN/Meganet2.jl/examples/ECNN_compareCIFAR10.mat', 'Y0','C','theta','Z','timeMAT');