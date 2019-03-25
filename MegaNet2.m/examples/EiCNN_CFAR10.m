clear;
addpath ../cifar-10-batches-mat/
addpath(genpath(pwd))

%useGPU = 1; precision='single';
useGPU = 0; precision='double';
%useGPU = 0; precision='single';

MNIST = 0;
if MNIST
    [Y0,C,Ytest,Ctest] = setupMNIST(2^12);
    Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
    nImg = [28 28];
    cin = 1;
else
    addpath ../../cifar-10-batches-mat/
    [Y0,C,Ytest,Ctest] = getCFAR10(2^6);
    nImg = [32 32];
    cin = 3;
end

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
Y0 = double(Y0);
Ytest = double(Ytest);
C= double(C);
Ctest = double(Ctest);
%% setup network
%act = @tanhActivation;
act = @reluActivation;
nf  = 16;

blocks    = cell(3,1); RegOps = cell(3,1);
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

nt = 25;
h  = 1e-1;
blocks{2} = iResNN(layerEXP,layerIMP,nt,h,'useGPU',useGPU,'precision',precision);
tt = nTheta(blocks{2});
RegOps{2} = opTimeDer(tt,blocks{2}.nt);

%% ResNet block
%sK   = [3,3,nf,nf];
%B = kron(speye(nf),ones(prod(nImg),1));
%layer = doubleSymLayer(convFFT(nImg,sK),'Bout',B*0);

%nt = 5;
%h  = 1e-1;
%blocks{2} = ResNN(layer,nt,h,'useGPU',useGPU,'precision',precision);
%tt = nTheta(blocks{2});
%RegOps{2} = opTimeDer(tt,blocks{2}.nt);

%% Close the network
cout      = 16; 
%blocks{3} = NN({singleLayer(convFFT(nImg,[3 3 nf cout]))},'useGPU',useGPU,'precision',precision);
%RegOps{3} = opEye(nTheta(blocks{3}));

%% Connector block
blocks{3} = connector(opPoolMCN([nImg cout],4));
RegOps{3} = opEye(nTheta(blocks{3}));

%% Put it all together
net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);

theta  = 0.01*max(randn(ntheta,1),0);
W = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));


RegOpW = blkdiag(opGrad(nImg/4,cout*10,ones(2,1)),opEye(10));
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
opt.learningRate = 0.01;
opt.maxEpochs    = 40;
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = 64;
opt.momentum     = 0.9;
opt.out          = 1;
    
%% run optimization
[xOpt,His1] = solve(opt,fctn,[theta(:); W(:)],fval);
opt.learningRate=opt.learningRate/10;
opt.maxIter = 20;
[xOpt,His2] = solve(opt,fctn,xOpt,fval);
[thOpt,WOpt] = split(fctn,xOpt);
