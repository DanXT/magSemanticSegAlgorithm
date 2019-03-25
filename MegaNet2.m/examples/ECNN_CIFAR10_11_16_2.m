clear all; clc;

[Y0,C,Ytest,Ctest] = getCFAR10(64);

nImg = [32 32]; cin =3;

useGPU = 0; precision='single';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);

%% setup network
act = @reluActivation;
conn_act = @identityActivation;
id_act = @identityActivation;
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv = @convMCN;
end

nt = 8;
h = 0.1;


cin_k = cin;
nImg_k = nImg;
blocks    = cell(0,1); RegOps = cell(0,1);


cout_k = 16;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k),1)));
blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k]),'Bin',B,'activation', id_act)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));


cin_k = cout_k;
% kernel1   = doubleLayer(conv(nImg_k,[3 3 cin_k cin_k]),conv(nImg_k,[3 3 cin_k cin_k]),'activation1',act,'activation2',id_act,'Bin1',B,'Bin2',B);
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'activation',act,'Bin',B);
blocks{end+1} = ResNN(kernel1,nt,h,'useGPU',useGPU,'precision',precision);
% RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);
RegOps{end+1} = opEye(nTheta(blocks{end}));

%%% DOWNSAMPLE AND EXPAND OPTION NO 1: %%%%%%%%%%%%%
blocks{end+1} = connector(opPoolMCN([nImg_k cin_k],2));
RegOps{end+1} = opEye(nTheta(blocks{end}));
nImg_k = nImg_k/2;

cout_k = 2*cin_k;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k),1)));
blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k]),'Bin',B,'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cin_k = cout_k;
% kernel1   = doubleLayer(conv(nImg_k,[3 3 cin_k cin_k]),conv(nImg_k,[3 3 cin_k cin_k]),'activation1',act,'activation2',id_act,'Bin1',B,'Bin2',B);
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'activation',act,'Bin',B);
blocks{end+1} = ResNN(kernel1,nt,2*h,'useGPU',useGPU,'precision',precision);
% RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{k}.nt);
RegOps{end+1} = opEye(nTheta(blocks{end}));


%%% DOWNSAMPLE AND EXPAND OPTION NO 2:
blocks{end+1} = connector(opPoolMCN([nImg_k cin_k],2));
RegOps{end+1} = opEye(nTheta(blocks{end}));
nImg_k = nImg_k/2;

cout_k = 2*cin_k;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k),1)));
blocks{end+1} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k]),'Bin',B,'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'activation',act,'Bin',B);
% kernel1   = doubleLayer(conv(nImg_k,[3 3 cin_k cin_k]),conv(nImg_k,[3 3 cin_k cin_k]),'activation1',act,'activation2',id_act,'Bin1',B,'Bin2',B);
blocks{end+1} = ResNN(kernel1,nt,4*h,'useGPU',useGPU,'precision',precision);
% RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);
RegOps{end+1} = opEye(nTheta(blocks{end}));

blocks{end+1} = connector(opPoolMCN([nImg_k cin_k],2));
RegOps{end+1} = opEye(nTheta(blocks{end}));
nImg_k = nImg_k/2;

cout = cout_k;
%%

net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);

theta  = 0.01*initTheta(net);
W = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));

RegOpW = blkdiag(opGrad(nImg_k,cout*10,ones(2,1)),opEye(10));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,1e-4,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,1e-4,[],'useGPU',useGPU,'precision',precision);

fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',512,'useGPU',useGPU,'precision',precision);
[theta,W] = gpuVar(useGPU,precision,theta,W);
%%

opt4 = sgd();
opt4.learningRate = 0.02;
opt4.maxEpochs = 150;
opt4.nesterov = true;
opt4.ADAM = false;
opt4.miniBatch = 64;
opt4.momentum = 0.9;
opt4.out = 1;



th0 = [theta(:); W(:)];

th = th0;
disp(['*************************** sample size = ', num2str(samplesize), '********************************************************************']);
batchsize = fval.batchSize;
% fctn =  dnnVarProBatchObjFctn(net,pRegKb,pLoss,pRegW,newtInner,Y0(:,1:samplesize),C(:,1:samplesize),'batchSize',batchsize,'useGPU',useGPU,'precision',precision);
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',batchsize,'useGPU',useGPU,'precision',precision);

    
[th,HIS] = solve(opt4,fctn,th,fval);
