clear all; clc;
[Y0,C,Ytest,Ctest] = setupMNIST(100);
Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
nImg = [28 28];

%% setup network
act = @reluActivation;

blocks    = cell(3,1); RegOps = cell(2,1);
blocks{1} = NN({singleLayer(convMCN(nImg,[3 3 1 4]),'activation', act)});
RegOps{1} = opEye(nTheta(blocks{1}));

kernel1   = doubleSymLayer(@tanhActivation, convMCN(nImg,[3 3 4 4]), []);
blocks{2} = LeapFrogNN(kernel1,5,.001);
% egOps{2} = opEye(nTheta(blocks{2}));
RegOps{2} = opTimeDer(nTheta(blocks{2}),blocks{2}.nt,.01);

blocks{3} = connector(getAveragePooling([nImg,4],1),[]);
%%


net    = MegaNet(blocks);
ntheta = nTheta(net);
theta  = initTheta(net);

pLoss = softmaxLoss();
W = vec(randn(10,nFeatOut(net)+1));

RegOpW = blkdiag(opGrad(nImg/2,40,ones(2,1)),opEye(10));
pRegW  = tikhonovReg(RegOpW,1e0);
pRegKb    = tikhonovReg(blkdiag(RegOps{:}),1e-4);

fctn = dnnObjFctn(net,pRegKb,pLoss,pRegW,Y0,C);
fval = dnnObjFctn(net,[],pLoss,[],Ytest,Ctest);
% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
% checkDerivative(fctn,[Kb(:);W(:)])
%%
opt1  = newton('out',1,'maxIter',20);
opt1.linSol = steihaugPCG('tol',1e-1,'maxIter',5);
opt2  = sd('out',1,'maxIter',20);
opt3  = nlcg('out',1,'maxIter',50);
 [KbWopt1,His1] = solve(opt1,fctn,[theta(:); W(:)],fval);
% [KbWopt2,His2] = solve(opt2,fctn,[Kb(:); W(:)],fval);
%  [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);
%%
[thetaOpt,WOpt] = split(fctn,KbWopt1);
figure(1); clf;
montageArray(reshape(WOpt(1:end-10),[nImg/2 40]));