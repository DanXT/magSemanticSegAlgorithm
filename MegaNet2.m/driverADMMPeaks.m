close all; clear all;

[Ytrain,Ctrain] = setupPeaks(1000,5);
Ytrain = Ytrain';
Ctrain = Ctrain';
% Ctrain = Ctrain(1,:);
% Cv = Cv(1,:);
% Ctrain = 0*/Ctrain;
% Ctrain(1,:) = Ytrain(1,:)>0;
% Ctrain(2,:) = Ytrain(1,:)<0;

% Ytrain = Ytrain(:,150:end);
% Ctrain = Ctrain(:,150:end);

% Ctrain = Ctrain(1,:);
% Cv     = Cv(1,:);

 figure;
 viewFeatures2D(Ytrain,Ctrain)

%% setup network
K     = dense([2,2]);
%  layer = doubleSymLayer(@identityActivation,K,[],'B2',eye(2));
 layer = singleLayer(K,'Bout',eye(2));
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
 pLoss = softmaxLoss();
%  pLoss = logRegressionLoss();
%% solve the coupled problem
regOp = opTimeDer(nTheta(net),nt,h);
 regOp = opEye(nTheta(net));
pRegK = tikhonovReg(regOp,1e-2,[]);
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
Z = Ytrain;
YN = apply(net,theta,Ytrain);
U = Z-Ytrain;
W= rand(size(Ctrain,1)*(size(YN,1)+1),1);
k=1
%%
for k=1 
    %% ADMM Step 1: Get W
    classSolver.out=2;
    fclass = classObjFctn(pLoss,pRegW,reshape(Z,nfeat,[]),Ctrain);
    W      = solve(classSolver,fclass,W);
    [F1,para] = eval(fclass,W)
    figure(k); clf
    
    subplot(1,3,1);
    viewFeatures2D(reshape(Z,nfeat,[]),Ctrain)
    hold on;
    tt = linspace(-2,2,101);
        plot(tt, -W(1)/W(2)*tt-W(3)/W(2),'-k','LineWidth',2);
%         axis([-2 2 -2 2])
%         title(sprintf('initial, err=%1.2e',trainErr));
        
    %% ADMM Step 2: Update Z
    pRegZ = tikhonovReg(opEye(numel(Z)),rho);
    pRegZ.xref = YN-U;
    szW = [size(Ctrain,1), size(YN,1)+1];
    fclassZ = classObjFctnZ(pLoss,pRegZ,reshape(W,szW),Ctrain);
    Z      = solve(classSolver,fclassZ,Z(:));
    [F2,para] = eval(fclassZ,Z);
    figure(k); 
    subplot(1,3,2);
    viewFeatures2D(reshape(Z,nfeat,[]),Ctrain)
%     axis([-2 2 -2 2]);
    %% ADMM Step 3: Update K
    ftheta = dnnADMMBatchObjFctn(net,pRegK, Ytrain, reshape(Z,nfeat,[]) +U);
    ftheta.batchSize=1000;
    if norm(theta)==0;
        theta = randn(size(theta));
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
    U = U + (reshape(Z,nfeat,[]) - YN);
    
%     pause;
end
