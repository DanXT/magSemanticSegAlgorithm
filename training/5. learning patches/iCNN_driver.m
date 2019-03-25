clc;clear all;close all;


if ispc  % true for windows platforms
    currentPath = '..\..\MegaNet2.m\';
    addpath(genpath(currentPath))
    addpath('functions\');
    useGPU = 0; precision='double';
    % load training data
    load('..\2. prepare patches for learning\result\trainingData.mat');      
else
    currentPath = '../../MegaNet2.m/';
    addpath(genpath(currentPath))
    addpath('functions/');
    useGPU = 0; precision='double';
    % load training data
    load('../2. prepare patches for learning/result/trainingData.mat'); 
end


[Y0,C] = gpuVar(useGPU,precision,Y0,C);
Y0 = double(Y0);
C= double(C);

% load test Data
if ispc
    load('..\4. prepare patches for test\result\testPatches1.mat');
else
    load('../4. prepare patches for test/result/testPatches1.mat');
end

cin = 6;
nImg = [64 64];
nClass = 3;
%% setup network
%act = @tanhActivation;
act = @reluActivation;
nf  = 32;

blocks    = cell(2,1); RegOps = cell(2,1);

%% Block to open the network
blocks{1} = NN({singleLayer(convFFT(nImg,[3 3 cin nf]),'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{1} = opEye(nTheta(blocks{1}));


%% iResNet block
% implicit layer
sKi      = [3,3,nf];
layerIMP = linearNegLayer(convBlkDiagFFT(nImg,sKi));
% explicit layer
sKe   = [1,1,nf,nf];
B = kron(speye(nf),ones(prod(nImg),1));
layerEXP = doubleSymLayer(convFFT(nImg,sKe),'Bout',B*0);

nt = 2;
h  = 1e-1;
blocks{2} = iResNN(layerEXP,layerIMP,nt,h,'useGPU',useGPU,'precision',precision);
tt = nTheta(blocks{2});
RegOps{2} = opTimeDer(tt,blocks{2}.nt);


%% Close the network
cout      = nf; 

%% Put it all together
net    = MegaNet(blocks);
pLoss = softmaxSegLoss('nClass',nClass, 'addWeights', 0);
ntheta = nTheta(net);

theta  = 0.01*max(randn(ntheta,1),0);
W = 0.01*vec(max(randn(nClass,cout+1),0));
RegOpW = blkdiag(opEye(length(W)));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,1e-4,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,1e-4,[],'useGPU',useGPU,'precision',precision);

%% Prepare optiization
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);

opt = sgd();
opt.learningRate = 0.0001;
opt.maxEpochs    = 20;
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = 64;
opt.momentum     = 0.9;
opt.out          = 1;
    

%% run optimization
[xOpt,His1] = solve(opt,fctn,[theta(:); W(:)],fval);
opt.learningRate=opt.learningRate/10;
opt.maxEpochs = 10;
[xOpt,His2] = solve(opt,fctn,xOpt,fval);
[thOpt,WOpt] = split(fctn,xOpt);

if ispc

    save('result\weightsOpt.mat', 'thOpt', 'WOpt', 'net','pLoss')

else 
    save('result/weightsOpt.mat', 'thOpt', 'WOpt', 'net','pLoss')
end







