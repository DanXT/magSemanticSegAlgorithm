close all; clear all;

% [Ytrain,Ctrain] = setupPeaks(1000,5);
[Ytrain,Ctrain,Yv,Cv] = setupBox('ntrain',500,'nval',400);

% Ytrain = Ytrain';
% Ctrain = Ctrain';
% Ctrain = Ctrain(1,:);
% Cv = Cv(1,:);
% Ctrain = 0*/Ctrain;
% Ctrain(1,:) = Ytrain(1,:)>0;
% Ctrain(2,:) = Ytrain(1,:)<0;

% Ytrain = Ytrain(:,150:end);
% Ctrain = Ctrain(:,150:end);

Ctrain = Ctrain(1,:);
 Cv     = Cv(1,:);

 figure;
 viewFeatures2D(Ytrain,Ctrain)

%% setup network
K     = dense([2,2]);
  layer = doubleSymLayer(K,'Bout',eye(2));
% layer.activation=@identityActivation;
% tY = linspace(0,10,40);
% tth = linspace(0,10,11);
% net = ResNNrk4(layer,tth,tY);
% nt = numel(tth);
% h  = tth(2)-tth(1);
 net   = ResNN(layer,100,20/100);
% net   = LeapFrogNN(layer,100,20/100);
nt = net.nt;
h = net.h;

%% setup classifier
%  pLoss = softmaxLoss();
pLoss = logRegressionLoss();
%% solve the coupled problem
regOp = opTimeDer(nTheta(net),nt,h);
%  regOp = opEye(nTheta(net));
pRegK = tikhonovReg(regOp,1e-8,[]);
regOpW = opEye((nFeatOut(net)+1)*size(Ctrain,1));
pRegW = tikhonovReg(regOpW,1e-2);
classSolver = newton();
classSolver.maxIter=5;
classSolver.linSol.maxIter=20;
classSolver.atol=1e-10
 opt = newton();
opt.out=2;
opt.atol=1e-16;
 opt.linSol.maxIter=20;
opt.maxIter=200;
opt.LS.maxIter=30;
% fctn = dnnVarProObjFctn(net,pRegK,pLoss,pRegW,classSolver,Ytrain,Ctrain);
% fval = dnnVarProObjFctn(net,[],pLoss,[],classSolver,Yv,Cv);


%% ADMM init
nfeat = size(Ytrain,1);
rho = 1e-6;
theta = zeros(nTheta(net),1);
Z = Ctrain;
YN = apply(net,theta,Ytrain);
 W= zeros(size(Ctrain,1)*(size(YN,1)),1);
% U = Z-Ytrain;
U = 0
k=1
%%
for k=1
    %% ADMM Step 1: Get Z
    Ye = speye(size(Ctrain,2));
    classSolver.out=2;
    pLossZ = logRegressionLossNoBias();
    pRegZ  = tikhonovReg(opEye(size(Ctrain,2)),1e-4,vec(YN'*W-U));
    fclass = classObjFctn(pLossZ,pRegZ,Ye,Ctrain);
    Z      = solve(classSolver,fclass,Z(:));
    [F1,para] = eval(fclass,Z)
    
    %% ADMM Step 2: get W
    W = blockLSQR(YN',Z-U,100,[],W);
    
    figure(k); clf
    subplot(1,3,1);
    viewFeatures2D(YN,Ctrain)
    hold on;
    tt = linspace(-2,2,101);
        plot(tt, -W(1)/W(2)*tt,'-k','LineWidth',2);
%         axis([-2 2 -2 2])
%         title(sprintf('initial, err=%1.2e',trainErr));
        
   
%     axis([-2 2 -2 2]);
    %% ADMM Step 3: Update K
    ftheta = dnnADMMBatchObjFctnEldad(net,pRegK, W,Ytrain,(Z +U)');
%     ftheta.Ytarget = C-.5;
    %%
    ftheta.batchSize=10000;
     if norm(theta)==0;
        theta = 1e-1*randn(size(theta));
     end
    theta = solve(opt,ftheta,theta);
    YN = apply(net,theta,Ytrain);
    %%
    figure(k);
    subplot(1,3,3);
%     YNv = apply(net,theta,Yv);
    viewFeatures2D(YN,Ctrain);
%     axis([-2 2 -2 2]);
    
    %% ADMM Step 4: Update U
     U = U + (YN'*W-Z);
    
%     pause;
end
