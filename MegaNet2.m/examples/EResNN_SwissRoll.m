close all; clear all;

[Ytrain,Ctrain,Yv,Cv] = setupSwissRoll(800);
Ctrain = Ctrain(1,:); 
Cv = Cv(1,:);
rng(20)
figure(1); clf;
subplot(1,3,1);
viewFeatures2D(Ytrain,Ctrain)
title('input features');
%% setup network
T =.1;
nt = 8;
K     = dense([8,2]);
 layer = doubleSymLayer(K,'B2',ones(8,1));
%  layer = singleLayer(K,'Bout',ones(2,1));
 tY = linspace(0,T,50);
 tth = linspace(0,T,nt+1);
 net = ResNNrk4(layer,tth,tY);
nt = numel(tth);
h  = tth(2)-tth(1);
% net   = ResNN(layer,nt,T/nt);
% nt = net.nt;
% h = net.h;

%% setup classifier
pLoss = softmaxLoss();
pLoss = logRegressionLoss();
%% solve the coupled problem
regOp = opTimeDer(nTheta(net),nt,h);
pRegK = tikhonovReg(regOp,1e-8,[]);
regOpW = opEye((nFeatOut(net)+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-2);

classSolver = newton();
classSolver.maxIter=4;
classSolver.linSol.maxIter=4;
opt = newton();
opt.out=2;
opt.atol=1e-16;
opt.linSol.maxIter=20;
opt.maxIter=30;
opt.LS.maxIter=20;
fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain);
fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);


% th0 = 1e0*max(randn(nTheta(net),1),0);
th0 = repmat(1e0*randn(nTheta(net.layer),1),nt,1);
%  W0  = randn((nDataOut(net)+1)*size(Ctrain,1),1);
% W0  = [1;0;0;0;1;0]
thetaOpt = solve(opt,fctn,th0,fval);
[Jc,para] = eval(fctn,thetaOpt);
WOpt = para.W;
    %%
[Ydata,Yn,tmp] = apply(net,thetaOpt,Yv);
figure(1);
subplot(1,3,2);
viewFeatures2D(Yn,Cv);
% title('output features')
subplot(1,3,3);
viewContour2D([-2 2 -1.5 1.5],thetaOpt,WOpt,net,pLoss);
hold on
viewFeatures2D(Yv,Cv);
return;
%%
