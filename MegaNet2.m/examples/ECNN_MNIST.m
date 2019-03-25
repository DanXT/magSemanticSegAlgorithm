clear all; clc;
[Y0,C,Ytest,Ctest] = setupMNIST(100);
Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
nImg = [28 28];

useGPU = 0; precision='single';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
%% setup network
act = @tanhActivation;

blocks    = cell(3,1); RegOps = cell(3,1);
blocks{1} = NN({singleLayer(convMCN(nImg,[3 3 1 2]),'activation', act)});
RegOps{1} = opEye(nTheta(blocks{1}));

kernel1   = singleLayer(convMCN(nImg,[3 3 2 2]),'activation', act);
blocks{2} = ResNN(kernel1,3,.1);
RegOps{2} = opTimeDer(nTheta(blocks{2}),blocks{2}.nt);

blocks{3} = NN({singleLayer(convMCN(nImg,[3 3 2 4],'stride',2),'activation', act)});
% blocks{3} = connector(opPoolMCN([nImg 2],2));
RegOps{3} = opEye(nTheta(blocks{3}));

kernel2   = singleLayer(convMCN(nImg/2,[3 3 4 4]),'activation', act);
blocks{4} = ResNN(kernel2,3,.1);
RegOps{4} = opTimeDer(nTheta(blocks{4}),blocks{4}.nt);
% RegOps{4} = opEye(nTheta(blocks{4}));

blocks{5} = NN({singleLayer(convMCN(nImg/2,[3 3 4 4],'stride',2),'activation', act)});
% blocks{5} = connector(opPoolMCN([nImg/2 2],2));
RegOps{5} = opEye(nTheta(blocks{5}));

net    = MegaNet(blocks);
pLoss = softmaxLoss();
ntheta = nTheta(net);
theta  = initTheta(net)/1e1;
W = 1e-6*vec(randn(10,nFeatOut(net)+1));


RegOpW = blkdiag(opGrad(nImg/4,40,ones(2,1)),opEye(10));
pRegW  = tikhonovReg(RegOpW,1e-4);
pRegKb = tikhonovReg(blkdiag(RegOps{:}),1e-4);

fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',10,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',10,'useGPU',useGPU,'precision',precision);
[theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
% checkDerivative(fctn,[Kb(:);W(:)])
%%
thetaW0 = [theta(:); W(:)];

opt1  = newton('out',1,'maxIter',20);
opt2  = sd('out',1,'maxIter',20);
opt3  = nlcg('out',1,'maxIter',50);
 [KbWopt1,His1] = solve(opt3,fctn,[theta(:); W(:)],fval);
% [KbWopt2,His2] = solve(opt2,fctn,[Kb(:); W(:)],fval);
% [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);