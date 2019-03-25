clear all; clc;
[Y0,C,Ytest,Ctest] = setupMNIST(1000);
Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
nImg = [28 28];

useGPU = 0; precision='double';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
%% setup network
act = @reluActivation;

nt = 20; 
h = .05;
layer= doubleSymLayer(convMCN(nImg,[3 3 1 1]),'activation',act);
outTimes = zeros(nt,1); outTimes(2:2:nt) = 1;
net = ResNN(layer,nt,h,'outTimes',outTimes);
RegOp = blkdiag(opTimeDer(nTheta(net),nt));

pLoss = softmaxLoss();
ntheta = nTheta(net);
theta  = initTheta(net)/1e0;

RegOpW = blkdiag(opGrad(nImg/4,nnz(outTimes)*10,ones(2,1)),opEye(10));
pRegW  = tikhonovReg(RegOpW,1e-3);
pRegKb = tikhonovReg(RegOp,1e-4);

 Q1 = kron(speye(nnz(outTimes)),getAveragePooling([nImg 1],2));
%
Q = opPoolMCN([nImg 1],4);
net.Q = Q;
cSolver = newton('maxIter',5);
fctn = dnnVarProBatchObjFctn(net,pRegKb,pLoss,pRegW,cSolver,Y0,C,'batchSize',500,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',500,'useGPU',useGPU,'precision',precision);
W = vec(randn(10,size(Q,1)+1));
[theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
% checkDerivative(fctn,[Kb(:);W(:)])
%%
thetaW0 = [theta(:); W(:)];

%%
[~,~,tmp] = net.apply(theta,Y0(:,end-2));
YN = cell2mat(tmp(outTimes==1,1));
figure(1); clf;
montageArray(reshape(YN,[nImg nnz(outTimes)]))
%%
%    opt1  = newton('out',1,'maxIter',5);
  opt2  = sd('out',1,'maxIter',20);
%   opt3  = nlcg('out',1,'maxIter',50);

 [KbWopt1,His1] = solve(opt2,fctn,theta(:),fval);
 
%  [KbWopt2,His2] = solve(opt2,fctn,[Kb(:); W(:)],fval);
% [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);