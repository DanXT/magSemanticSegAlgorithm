clear;

% normalize examples after each block
doInstNorm = 1;


% useGPU = 1; precision='single';
useGPU = 0; precision='double';
%useGPU = 0; precision='single';

% [Y0,C,Ytest,Ctest ] = setupSTL(5000,8000);
[Y0,C,Ytest,Ctest ] = setupSTL(50,80);
nImg = [96 96];
C = C'; Ctest = Ctest';
cin = 3;

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
if strcmp(precision,'double');
    Y0 = double(Y0);
    Ytest = double(Ytest);
    C= double(C);
    Ctest = double(Ctest);
end
%% setup network
%act = @tanhActivation;
act = @reluActivation;
nf  = 8;

blocks    = cell(0,1); RegOps = cell(0,1);
%% Block to open the network
B = kron(eye(nf),ones(prod(nImg),1));

blocks{end+1} = NN({singleLayer(convMCN(nImg,[3 3 cin nf]),'activation', act,'Bin',B)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
if doInstNorm
    blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
end

%% iResNet block
% implicit layer
sKi      = [3,3,nf];
layerIMP = linearNegLayer(convBlkDiagFFT(nImg,sKi));
% explicit layer
sKe   = [1,1,nf,nf];
layerEXP = singleLayer(convMCN(nImg,sKe),'activation',act,'Bout',B);

nt = 8;
h  = 1e-1;
blocks{end+1} = iResNN(layerEXP,layerIMP,nt,h,'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);
if doInstNorm
    blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
end

%% Connector block
blocks{end+1} = connector(B'/size(B,1));
RegOps{end+1} = opEye(nTheta(blocks{end}));

%% Put it all together
net    = MegaNet(blocks);
pLoss = softmaxLoss();

theta  = initTheta(net);
W = 0.1*randn(10,nFeatOut(net)+1);


RegOpW = blkdiag(opEye(10*(nFeatOut(net)+1)));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,1e-1,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,1e-1,[],'useGPU',useGPU,'precision',precision);

%% Prepare optimization
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);

opt = sgd();
if doInstNorm
    opt.learningRate = @(epoch) 1e-2/sqrt(epoch);
else
    opt.learningRate = 1e-2;
end
opt.maxEpochs    = 60;
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = 64;
opt.momentum     = 0.9;
opt.out          = 1;

%% run optimization
alphaW = logspace(-3,0,4);
alphaT = logspace(-3,0,4);
for i=1:numel(alphaW)
	fctn.pRegW.alpha = alphaW(i);
	for j=1:numel(alphaT)
		fctn.pRegTheta.alpha = alphaT(j);
		[xOpt,His] = solve(opt,fctn,[theta(:); W(:)],fval);
		[thOpt,WOpt] = split(fctn,gather(xOpt));
		save(sprintf('EiCNN_avg_%d_%d_STL10_aW_%1.1e_aT_%1.1e.mat',nt,nf,fctn.pRegW.alpha,fctn.pRegTheta.alpha),'thOpt','WOpt','His')
	end
end

