clear all; clc;
[Y0,C,Ytest,Ctest] = setupMNIST(400);
Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
nImg = [28 28];

% useGPU = 1; precision='single';
useGPU = 0; precision='double';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
%% setup network
act = @tanhActivation;

blocks    = cell(3,1); RegOps = cell(3,1);
blocks{1} = NN({singleLayer(convMCN(nImg,[3 3 1 2]),'activation', act)});
RegOps{1} = opEye(nTheta(blocks{1}));

kernel1   = singleLayer(convMCN(nImg,[3 3 2 2]),'activation', act);
blocks{2} = ResNN(kernel1,3,.1);
RegOps{2} = opTimeDer(nTheta(blocks{2}),blocks{2}.nt);

blocks{3} = connector(opPoolMCN([nImg 2],2));
RegOps{3} = opEye(nTheta(blocks{3}));

net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);
theta  = initTheta(net);
W = vec(randn(10,nFeatOut(net)+1));


RegOpW = blkdiag(opGrad(nImg/2,20,ones(2,1)),opEye(10));
pRegW  = tikhonovReg(RegOpW,1e-2);
pRegKb = tikhonovReg(blkdiag(RegOps{:}),1e-4);

newtInner =newton('out',0,'maxIter',10);
fctn = dnnVarProBatchObjFctn(net,pRegKb,pLoss,pRegW,newtInner,Y0,C,'batchSize',10,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',10,'useGPU',useGPU,'precision',precision);
[theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
% checkDerivative(fctn,[Kb(:);W(:)])
%%
thetaW0 = [theta(:); W(:)];

opt1  = newton('out',1,'maxIter',20);
opt2  = sd('out',1,'maxIter',20);
opt3  = nlcg('out',1,'maxIter',50);
[KbWopt1,His1] = solve(opt1,fctn,theta(:),fval);
%  [KbWopt2,His2] = solve(opt2,fctn,theta(:),fval);
%  [KbWopt3,His3] = solve(opt3,fctn,theta(:),fval);