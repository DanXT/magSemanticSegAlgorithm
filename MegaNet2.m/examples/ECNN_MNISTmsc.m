clear all; 
% clc;
[Y0,C,Ytest,Ctest] = setupMNIST(1000);
Y0 = Y0'; Ytest = Ytest'; C = C'; Ctest = Ctest';
nchannels = 1;
Y0 = repmat(Y0,nchannels,1);
Ytest = repmat(Ytest,nchannels,1);
nImg = [28 28];

useGPU = 0; precision='double';

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
%% setup network
act = @reluActivation;

nt = 50; 
h = .01;
layer= doubleSymLayer(convMCN(nImg,[3 3 nchannels nchannels]));
 outTimes = ones(nt,1);
%  outTimes(1)=1;
%   outTimes(end:-2:1) = 1;
  outTimes(10)=1
  outTimes(end)=1;
 net = ResNN(layer,nt,h,'outTimes',outTimes);
% net = LeapFrogNN(layer,nt,h,'outTimes',outTimes);
RegOp = blkdiag(opTimeDer(nTheta(net),nt));

pLoss = softmaxLoss();
ntheta = nTheta(net);
% th1 = [-1;0;-1;0;4;-1;0;-1;0]/8;
th0 = randn(9,nchannels^2);
th0 = (th0 - sum(th0,1))/9;
theta = repmat(th0(:),nt,1);


RegOpW = blkdiag(opGrad(nImg/4,nchannels*nnz(outTimes)*10,ones(2,1)),opEye(10));
pRegW  = tikhonovReg(RegOpW,1e-4);
pRegKb = tikhonovReg(RegOp,1e-4);

%  Q1 = kron(speye(nnz(outTimes)),getAveragePooling([nImg 2],2));
%
Q = opPoolMCN([nImg nchannels],4);
net.Q = Q;
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',200,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',200,'useGPU',useGPU,'precision',precision);
W = vec(randn(10,size(Q,1)*nnz(outTimes)+1))/1e2;
[theta,W] = gpuVar(fctn.useGPU,fctn.precision,theta,W);
% [Jc,para,dJ,H,PC] = fctn([Kb(:);W(:)]);
% checkDerivative(fctn,[Kb(:);W(:)])

%%
[Yd,~,tmp] = net.apply(theta,Y0(:,end-2));
Yimg = cell2mat(tmp(net.outTimes==1,1));

figure(1); clf;
montageArray(reshape(Yimg,[nImg nchannels*nnz(net.outTimes)]))

%% look at rank
[Yout] = net.apply(theta,Y0);
Youtv  = net.apply(theta,Ytest);
%%
cfctn   = classObjFctn(pLoss,pRegW,Yout,C);
cfctnv  = classObjFctn(pLoss,[],Youtv,Ctest);
pRegW.alpha = cfctn.pRegW.alpha;
optClass = newton('out',2,'maxIter',100);
Yb = [Yout; ones(1,size(Yout,2))];
% Wls = Yb'\C';
% norm(Yb'*Wls-C')
 %%
W0     = zeros(size(Yb,1),size(C,1));
W      = solve(optClass,cfctn,W0(:),cfctnv);
            
%%
%    opt1  = newton('out',1,'maxIter',5);
%   opt2  = sd('out',1,'maxIter',10);
 opt3  = nlcg('out',1,'maxIter',10);

 [KbWopt1,His1] = solve(opt3,fctn,[theta(:); W(:)],fval);
 
%  [KbWopt2,His2] = solve(opt2,fctn,[Kb(:); W(:)],fval);
% [KbWopt3,His3] = solve(opt3,fctn,[theta(:); W(:)],fval);
%%
[Kopt,Wopt] = split(fctn,KbWopt1);

[YN,~,tmp] = net.apply(Kopt,Y0(:,end-2));
Yi = cell2mat(tmp(outTimes==1,1));
figure(2); clf;
montageArray(reshape(Yi,[nImg nnz(outTimes)]))
%% look at SVD
YN = net.apply(Kopt,Y0);
YN = reshape(YN,[],size(Y0,2));
 [U,S,V] = svd(Q*Y0,'econ');
 figure(3); clf;
hold on;
semilogy(diag(S),'or')
%%
