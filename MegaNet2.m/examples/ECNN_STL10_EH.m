clear;
addpath ../cifar-10-batches-mat/
addpath(genpath(pwd))

%useGPU = 1; precision='single';
useGPU = 0; precision='double';
%useGPU = 0; precision='single';

[Y0,C,Ytest,Ctest] = getSTL10(4500);
nImg = [96 96];
cin = 3;

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
tvn = normLayer([prod(nImg),nf,size(Y0,2)],[0,1,0]);
blocks{1} = NN({singleLayer(convFFT(nImg,[3 3 cin nf]),'activation', act),tvn},'useGPU',useGPU,'precision',precision);
RegOps{1} = opEye(nTheta(blocks{1}));

%% ResNet block

sK      = [3,3,nf];
B = kron(speye(nf),ones(prod(nImg),1));
layer = doubleSymLayer(convCircFFT(nImg,sK),'Bout',B,'nLayer',tvn);

nt = 25;
h  = 1e-1;
blocks{2} = ResNN(layer,nt,h,'useGPU',useGPU,'precision',precision);
RegOps{2} = opTimeDer(nTheta(blocks{2}),blocks{2}.nt);
    
%% Close the network
cout      = nf; 

%% Connector block
blocks{3} = connector(opPoolMCN([nImg cout],16));
RegOps{3} = opEye(nTheta(blocks{3}));

%% Put it all together
net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);

theta  = 0.01*max(randn(ntheta,1),0);
W = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));

RegOpW = blkdiag(opGrad(nImg/16,cout*10,ones(2,1)),opEye(10));
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
opt.learningRate = 1e-2;
opt.maxEpochs    = 40;
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = 1024;
opt.momentum     = 0.9;
opt.out          = 1;
    
%% run optimization
[xOpt,His1] = solve(opt,fctn,[theta(:); W(:)],fval);
opt.learningRate=opt.learningRate/10;
opt.maxEpochs = 20;
[xOpt,His2] = solve(opt,fctn,xOpt,fval);
[thOpt,WOpt] = split(fctn,xOpt);
