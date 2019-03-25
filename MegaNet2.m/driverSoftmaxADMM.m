close all; clear all;

% [Ytrain,Ctrain] = setupPeaks(1000,5);
[Ytrain,Ctrain,Yv,Cv] = setupBox('ntrain',500,'nval',400);

Ctrain = zeros(2,size(Ytrain,2));
Ctrain(1,:) = Ytrain(1,:)>0;
Ctrain(2,:) = Ytrain(1,:)<0;

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
%% setup classifier
%   pLoss = softmaxLoss();
 pLoss = logRegressionLoss();
%% solve the coupled problem
% regOpW = opEye((nFeatOut(net)+1)*size(Ctrain,1));
% pRegW = tikhonovReg(regOpW,1e-2);

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
Z = Ctrain;
YN = Ytrain;
W= zeros(size(Ctrain,1)*(size(YN,1)),1);
U = 0;
k=1

%%
his = []; Wall = [];
for k=1:100
    %% ADMM Step 1: Get Z
    Ye = speye(size(Ctrain,2));
    pLossZ = logRegressionLossNoBias()
%     LossNoBias();
    pRegZ  = tikhonovReg(opEye(size(Ctrain,2)),1e-1,vec(YN'*W+U));
    fclass = classObjFctn(pLossZ,pRegZ,Ye,Ctrain);
    Z      = solve(opt,fclass,Z(:));
    [F1,para] = eval(fclass,Z);
    
    %% ADMM Step 2: get W
%     W = blockLSQR(YN',Z-U,100,[],W);
    W = YN'\(Z-U);
    Wall = [Wall, W];
    figure(1); clf
    subplot(1,3,1);
    viewFeatures2D(YN,Ctrain)
    hold on;
    tt = linspace(-2,2,101);
        plot(tt, -W(1)/W(2)*tt,'-k','LineWidth',2);
%         axis([-2 2 -2 2])
%         title(sprintf('initial, err=%1.2e',trainErr));
        
   
%     axis([-2 2 -2 2]);
    
    %% ADMM Step 4: Update U
     U = U + (YN'*W-Z);
    his = [his;norm(YN'*W-Z)];
    
%     pause;
end
%%
Wtrue = YN'\vec(Ctrain);

figure(1); clf;
viewFeatures2D(YN,Ctrain);
hold on;
tt = linspace(-2,2,101);
plot(tt, -Wtrue(1)/Wtrue(2)*tt,'--k','LineWidth',2);
hold on;
plot(tt, -Wall(1,end)/Wall(2,end)*tt,'-r','LineWidth',2);
axis equal
plot(tt, -Wall(1,1)/Wall(2,1)*tt,'-k','LineWidth',2);
axis([-2 2 -2 2])