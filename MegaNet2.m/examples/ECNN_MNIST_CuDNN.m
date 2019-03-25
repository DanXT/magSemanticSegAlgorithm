clear all; 
[Y0,C,Ytest,Ctest] = setupMNIST(512);
Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
nImg = [28 28];

% Y0 = single(Y0'); Ytest = single(Ytest'); C = single(C'); Ctest = single(Ctest');

cudnnSession = convCuDNN2DSession();
% 
% cudnnSession = [];
Y0    = gpuArray(single(Y0));
C     = gpuArray(single(C));
Ytest = gpuArray(single(Ytest));
Ctest = gpuArray(single(Ctest));



%% setup network
act = @tanhActivation;

blocks    = cell(3,1); RegOps = cell(3,1);
blocks{1} = NN({singleLayer(convCuDNN2D(cudnnSession,nImg,[3 3 1 8]),'activation', act)});
RegOps{1} = opEye(nTheta(blocks{1}));

kernel1   = singleLayer(convCuDNN2D(cudnnSession,nImg,[3 3 8 8]),'activation', act);
blocks{2} = ResNN(kernel1,10,.05);
%  RegOps{2} = opTimeDer(nTheta(blocks{2}),blocks{2}.nt);
RegOps{2} = opEye(nTheta(blocks{2}));

blocks{3} = NN({singleLayer(convCuDNN2D(cudnnSession,nImg,[3 3 8 4],'stride',2),'activation', act)});
% blocks{3} = connector(opPoolMCN([nImg 2],2));
RegOps{3} = opEye(nTheta(blocks{3}));

kernel2   = singleLayer(convCuDNN2D(cudnnSession,nImg/2,[3 3 4 4]),'activation', act);
blocks{4} = ResNN(kernel2,20,.05);
RegOps{4} = opEye(nTheta(blocks{4}));

blocks{5} = NN({singleLayer(convCuDNN2D(cudnnSession,nImg/2,[3 3 4 4],'stride',2),'activation', act)});
% blocks{5} = connector(opPoolMCN([nImg/2 2],2));
RegOps{5} = opEye(nTheta(blocks{5}));

net    = MegaNet(blocks);
ntheta = nTheta(net);
theta  = gpuArray(single(initTheta(net)/1e2));

pLoss = softmaxLoss();
W = gpuArray(single(vec(randn(10,nFeatOut(net)+1))));

pRegW  = tikhonovReg(opEye(numel(W)),1e-4);
pRegKb    = tikhonovReg(blkdiag(RegOps{:}),1e-4);

% fctn = dnnObjFctn(net,pRegKb,pLoss,pRegW,Y0,C);
% fval = dnnObjFctn(net,[],pLoss,[],Ytest,Ctest);

fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',16);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',16);



% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
% checkDerivative(fctn,[Kb(:);W(:)])

opt1  = newton('out',1,'maxIter',10);
opt2  = sd('out',1,'maxIter',20);
opt3  = nlcg('out',1,'maxIter',20);
tic
 [KbWopt1,His1] = solve(opt1,fctn,[theta(:); W(:)],fval);
toc
clear cudnnSession;

 % [KbWopt2,His2] = solve(opt2,fctn,[Kb(:); W(:)],fval);
% [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);