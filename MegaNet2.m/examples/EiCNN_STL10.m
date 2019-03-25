clear;

% normalize examples after each block
doInstNorm = 1;
miniBatchSize = 16;

%useGPU = 1; precision='single';
%useGPU = 0; precision='double';
useGPU = 0; precision='single';

[Y0,C,Ytest,Ctest ] = setupSTL(2^7,2^7);
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
nf  = 6;

blocks    = cell(0,1); RegOps = cell(0,1);
%% Block to open the network
nL = nLayer([prod(nImg(1:2)) nf(1) miniBatchSize],'isWeight',1);
blocks{end+1} = NN({singleLayer(conv(nImg,[3 3 cin nf(1)]),...
                   'activation', act,'nLayer',nL)},...
                   'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));

%% iResNet block
% implicit layer
sKi      = [3,3,nf];
layerIMP = linearNegLayer(convBlkDiagFFT(nImg,sKi));
% explicit layer
sKe   = [1,1,nf,nf];
B = kron(speye(nf),ones(prod(nImg),1));
layerEXP = singleLayer(convMCN(nImg,sKe),'activation',act,'Bout',B);

nt = 5;
h  = 1e-1;
blocks{end+1} = iResNN(layerEXP,layerIMP,nt,h,'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);
if doInstNorm
    blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
end

%% Connector block
blocks{end+1} = connector(opPoolMCN([nImg nf],4));
RegOps{end+1} = opEye(nTheta(blocks{end}));

%% Put it all together
net    = MegaNet(blocks);
pLoss = softmaxLoss();

theta  = initTheta(net);
W = 0.01*vec(max(randn(10,nFeatOut(net)+1),0));


RegOpW = blkdiag(opGrad(nImg/4,nf*10,ones(2,1)),opEye(10));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,1e-4,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,1e-4,[],'useGPU',useGPU,'precision',precision);

%% Prepare optimization
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',256,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',256,'useGPU',useGPU,'precision',precision);

opt = sgd();
if doInstNorm
    opt.learningRate = @(epoch) 1e-2/sqrt(epoch);
else
    opt.learningRate = 1e-2;
end
opt.maxEpochs    = 100;
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = 64;
opt.momentum     = 0.9;
opt.out          = 1;

return
%% run optimization
[xOpt,His] = solve(opt,fctn,[theta(:); W(:)],fval);
[thOpt,WOpt] = split(fctn,gather(xOpt));


