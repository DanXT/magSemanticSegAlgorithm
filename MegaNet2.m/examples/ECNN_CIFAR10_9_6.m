clear all; clc;

[Y0,C,Ytest,Ctest] = getCFAR10(50);




nImg = [32 32]; cin =3;

useGPU = 0; precision='single';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
%% setup network
act = @reluActivation;
 conv = @convMCN;
% cudnnSession = convCuDNN2DSession();
% conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});

% conv = @(varargin)convMCN(varargin{:}); useGPU = 0; precision='single';


cin_k = cin;
nImg_k = nImg;
blocks    = cell(1,1); RegOps = cell(1,1);


k = 1;
cout_k = 16;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k),1)));
blocks{k} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k]),'Bout',B,'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'Bout',B,'activation',act);
blocks{k} = ResNN(kernel1,8,1,'useGPU',useGPU,'precision',precision);
RegOps{k} = opTimeDer(nTheta(blocks{k}),blocks{k}.nt);
% RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;

cout_k = 2*cin_k;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k/2),1)));
blocks{k} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k],'stride',2),'Bout',B,'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{k} = opEye(nTheta(blocks{k}));
nImg_k = nImg_k/2;
k = k+1;

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'Bout',B,'activation',act);
blocks{k} = ResNN(kernel1,8,1,'useGPU',useGPU,'precision',precision);
RegOps{k} = opTimeDer(nTheta(blocks{k}),blocks{k}.nt);
% RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;

cout_k = 2*cin_k;
B = gpuVar(useGPU,precision,kron(eye(cout_k),ones(prod(nImg_k/2),1)));
blocks{k} = NN({singleLayer(conv(nImg_k,[3 3 cin_k cout_k],'stride',2),'Bout',B,'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{k} = opEye(nTheta(blocks{k}));
nImg_k = nImg_k/2;
k = k+1;

cin_k = cout_k;
kernel1   = doubleSymLayer(conv(nImg_k,[3 3 cin_k cin_k]),'Bout',B,'activation',act);
blocks{k} = ResNN(kernel1,8,1,'useGPU',useGPU,'precision',precision);
RegOps{k} = opTimeDer(nTheta(blocks{k}),blocks{k}.nt);
% RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;


cout = cout_k;
%%

net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);
theta  = 0.01*max(randn(ntheta,1),0);%initTheta(net);
W = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));


RegOpW = blkdiag(opGrad(nImg_k,cout*10,ones(2,1)),opEye(10));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,2e-3,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,2e-3,[],'useGPU',useGPU,'precision',precision);

% fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',64,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',64,'useGPU',useGPU,'precision',precision);
[theta,W] = gpuVar(useGPU,precision,theta,W);
% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
%%
opt1  = newton('out',1,'maxIter',200);
opt1.linSol = steihaugPCG('tol',1e-2,'maxIter',50);
opt1.LS.maxIter =20;
opt2  = sd('out',1,'maxIter',500);
opt2.LS.maxIter =20;
opt3  = nlcg('out',1,'maxIter',500);
opt3.LS.maxIter =20;


opt4 = sgd();
opt4.learningRate = 0.1;
opt4.maxEpochs = 200;
opt4.nesterov = true;
opt4.miniBatch = 64;
opt4.momentum = 0.9;
opt4.out = 1;



newtInner  = newton('out',0,'maxIter',5);
newtInner.linSol = steihaugPCG('tol',1e-2,'maxIter',20);

% [KbWopt1,His1] = solve(opt1,fctn,theta(:),fval);


idKernel = zeros(3,3); idKernel(2,2)=1;
idK      = repmat([idKernel(:)],32,1);


% [TH,HIS] = deal(cell(1,1));
cnt = 1;
 %th0 = theta(:);
 th0 = [theta(:); W(:)];

th0(1:numel(idK)) = th0(1:numel(idK))+idK(:);


p = randperm(size(Y0,2));
Y0 = Y0(:,p);
C = C(:,p);
clear p;

th = th0;
% for jj = 5:ceil(log2(size(Y0,2)/1024))
    samplesize = min(50000,size(Y0,2)); %min(1024*(2^jj),size(Y0,2));
    disp(['*************************** sample size = ', num2str(samplesize), '********************************************************************']);
    batchsize = 32;
    % fctn =  dnnVarProBatchObjFctn(net,pRegKb,pLoss,pRegW,newtInner,Y0(:,1:samplesize),C(:,1:samplesize),'batchSize',batchsize,'useGPU',useGPU,'precision',precision);
    fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0(:,1:samplesize),C(:,1:samplesize),'batchSize',batchsize,'useGPU',useGPU,'precision',precision);
    tic
    [th,HIS{cnt}] = solve(opt4,fctn,th,fval);
    toc
%     TH{cnt}=th;

    cnt = cnt + 1;
% end
