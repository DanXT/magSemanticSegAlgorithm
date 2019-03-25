clear all;


useGPU = 1; precision='single';
% useGPU = 0; precision='double';



MNIST = 0;
if MNIST
    [Y0,C,Ytest,Ctest] = setupMNIST(2^15);
    Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
    nImg = [28 28];
    cin = 1;
else
    [Y0,C,Ytest,Ctest] = getCFAR10(2^14);
    nImg = [32 32];
    cin = 3;
end




cudnnSession = convCuDNN2DSession();
% cudnnSession = [];
conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});


[Y0,C] = gpuVar(useGPU,precision,Y0,C);
%% setup network
act = @tanhActivation;
% act = @reluActivation;

k = 1;
blocks    = cell(4,1); RegOps = cell(4,1);
blocks{k} = NN({singleLayer(conv(nImg,[3 3 cin 32]),'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;

B = gpuVar(useGPU,precision,kron(eye(32),ones(32*32,1)));
kernel1   = doubleSymLayer(act,conv(nImg,[3 3 32 32]),B);

blocks{k} = ResNN(kernel1,24,0.1,'useGPU',useGPU,'precision',precision);
RegOps{k} = opTimeDer(nTheta(blocks{k}),blocks{k}.nt);
% RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;


% blocks{k} = connector(opPoolMCN([nImg 32],2));
% RegOps{k} = opEye(nTheta(blocks{k}));
% k = k+1;


blocks{k} = NN({singleLayer(conv(nImg,[3 3 32 32],'stride',2),'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;


blocks{k} = NN({singleLayer(conv(nImg/2,[3 3 32 32],'stride',2),'activation', act)},'useGPU',useGPU,'precision',precision);
RegOps{k} = opEye(nTheta(blocks{k}));
k = k+1;

% kernel1   = singleLayer(convCuDNN2D(cudnnSession,nImg/2,[3 3 32 32]),'activation', act);
% blocks{k} = ResNN(kernel1,4,.1,'useGPU',useGPU,'precision',precision);
% RegOps{k} = opTimeDer(nTheta(blocks{k}),blocks{k}.nt);
% RegOps{k} = opTimeDer(nTheta(blocks{k}),blocks{k}.nt);
% k=k+1;


% blocks{k} = connector(opPoolMCN([nImg/2 64],2));
% RegOps{k} = opEye(nTheta(blocks{k}));
% k = k+1;

% blocks{k} = NN({singleLayer(convCuDNN2D(cudnnSession,nImg/4,[3 3 64 64]),'activation', act)},'useGPU',useGPU,'precision',precision);
% RegOps{k} = opEye(nTheta(blocks{k}));
% k=k+1;

cout = 32;


net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);
theta  = 0.01*max(randn(ntheta,1),0);%initTheta(net);
W = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));


RegOpW = blkdiag(opGrad(nImg/4,cout*10,ones(2,1)),opEye(10));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,1e-4,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,1e-4,[],'useGPU',useGPU,'precision',precision);

newtInner = newton('out',0,'maxIter',20);
newtInner.linSol =  steihaugPCG('tol',1e-1,'maxIter',50);

% 
% fctn = dnnVarProBatchObjFctn(net,pRegKb,pLoss,pRegW,newtInner,Y0,C,'batchSize',size(Y0,2),'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',64,'useGPU',useGPU,'precision',precision);

[theta,W] = gpuVar(useGPU,precision,theta,W);
%%
% thetaW0 = [theta(:); W(:)];

opt1  = newton('out',1,'maxIter',30);
opt1.linSol = steihaugPCG('tol',1e-1,'maxIter',20);
opt2  = sd('out',1,'maxIter',20);
opt3  = nlcg('out',1,'maxIter',100);


% [KbWopt1,His1] = solve(opt1,fctn,theta(:),fval);


p = randperm(size(Y0,2));
Y0 = Y0(:,p);
C = C(:,p);
clear p;



% th = theta(:);
th = [theta(:); W(:)];
for jj = 0:log2(size(Y0,2)/1024)
%     jj = 2;
    disp('***********************************************************************************************')
    samplesize = 1024*(2^jj);
    batchsize = 128;
    samplesize
    batchsize = 8;
%     fctn = dnnVarProBatchObjFctn(net,pRegKb,pLoss,pRegW,newtInner,Y0(:,1:samplesize),C(:,1:samplesize),'batchSize',batchsize,'useGPU',useGPU,'precision',precision);
    fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0(:,1:samplesize),C(:,1:samplesize),'batchSize',batchsize,'useGPU',useGPU,'precision',precision);
    tic
    th = solve(opt1,fctn,th,fval);
    toc
end








%  [KbWopt2,His2] = solve(opt2,fctn,theta(:),fval);
%  [KbWopt3,His3] = solve(opt3,fctn,theta(:),fval);