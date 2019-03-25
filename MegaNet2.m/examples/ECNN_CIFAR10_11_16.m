clear all; clc;

% get training and validation data
[Y0,C,Ytest,Ctest] = getCFAR10(512);

nImg = [32 32]; cin =3;

useGPU = 0; precision='single';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);

%% setup network
connAct = @identityActivation;
act     = @reluActivation;
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv    = @convMCN;
end

h = .1; 
nt = 8;

cin_k = cin;
nImg_k = nImg;
blocks    = cell(0,1); RegOps = cell(0,1);

cout_k = 16;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k),1)));

    blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k]),'Bout',B,'activation', connAct)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'Bout',B,'activation',act);
blocks{end+1} = ResNN(kernel1,nt,h,'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
% RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);

%%
cout_k = 2*cin_k;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k/2),1)));
blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k],'stride',2),'Bout',B,'activation', connAct)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
nImg_k = nImg_k/2;

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'Bout',B,'activation',act);
blocks{end+1} = ResNN(kernel1,nt,h,'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
% RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);

%%
cout_k = 2*cin_k;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k/2),1)));
blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k],'stride',2),'Bout',B,'activation', connAct)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
nImg_k = nImg_k/2;

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'Bout',B,'activation',act);
blocks{end+1} = ResNN(kernel1,nt,h,'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
% RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);

B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k/2),1)));
blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k],'stride',2),'Bout',B,'activation', connAct)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
nImg_k = nImg_k/2;
cout = cout_k;
%%

net    = MegaNet(blocks);
pLoss  = softmaxLoss();
ntheta = nTheta(net);
theta  = initTheta(net)/1e2;
W      = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));


RegOpW = blkdiag(opGrad(nImg_k,cout*10,ones(2,1)),opEye(10));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,2e0,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,2e0,[],'useGPU',useGPU,'precision',precision);

fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',512,'useGPU',useGPU,'precision',precision);
[theta,W] = gpuVar(useGPU,precision,theta,W);

opt4 = sgd();
opt4.learningRate = .05;
opt4.maxEpochs = 200;
opt4.nesterov = true;
opt4.ADAM=false;
opt4.miniBatch = 64;
opt4.momentum = 0.9;
opt4.out = 1;

cnt = 1;
th0 = [theta(:); W(:)];

th = th0;
samplesize = min(50000,size(Y0,2)); %min(1024*(2^jj),size(Y0,2));
disp(['*************************** sample size = ', num2str(samplesize), '********************************************************************']);
batchsize = fval.batchSize;
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0(:,1:samplesize),C(:,1:samplesize),'batchSize',batchsize,'useGPU',useGPU,'precision',precision);
tic
[th,HIS{cnt}] = solve(opt4,fctn,th,fval);
toc
