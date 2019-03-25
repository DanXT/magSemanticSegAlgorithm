clear;

% normalize examples after each block
doInstNorm = 1;
% loadTFinit = 1;
% tfinit = '/Users/lruthot/Dropbox/Projects/MegaNet2new/examples/EResNN_STL10-tf/init.mat';


% useGPU = 0; precision='double';
useGPU = 1; precision='single';
%useGPU = 0; precision='single';
pad = 8;
[Y0,C,Ytest,Ctest ] = setupSTLtf(50,50,pad);
nImg = [96 96]+2*pad;
% C = C'; Ctest = Ctest';
cin = 3;

[Y0,C] = gpuVar(useGPU,precision,Y0,C);
if strcmp(precision,'double')
    Y0 = double(Y0);
    Ytest = double(Ytest);
    C= double(C);
    Ctest = double(Ctest);
end
if useGPU
    cudnnSession = convCuDNN2DSession();
    conv = @(varargin)convCuDNN2D(cudnnSession,varargin{:});
else
    conv    = @convMCN;
end
%% setup network
act1 = @reluActivation;
actc = @reluActivation;
act = @reluActivation;

nf  = 16*[1;4;8;16;16];
nt  = 3*[1;1;1;1];
h  = 1*[1;1;1;1];

blocks    = cell(0,1); RegOps = cell(0,1);
%% Block to open the network
B = gpuVar(useGPU,precision,kron(eye(nf(1)),ones(prod(nImg),1)));
blocks{end+1} = NN({singleLayer(convMCN(nImg,[3 3 cin nf(1)]),'activation', act1,'Bin',B)},'useGPU',useGPU,'precision',precision);
RegOps{end+1} = opEye(nTheta(blocks{end}));
if doInstNorm
%     blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
    blocks{end+1} = NN({getBatchNormLayer([prod(nImg(1:2)) nf(1) 50], 'isWeight',1)});
	regD = gpuVar(useGPU,precision,zeros(nTheta(blocks{end}),1));
    RegOps{end+1} = opDiag(regD);

end
%% UNITs: iResNet blocks
for k=1:numel(h)
    nImgc = nImg/(2^(k-1));
    % implicit layer
    K = conv(nImgc,[3 3 nf(k) nf(k)]);
    B = gpuVar(useGPU,precision,kron(eye(nf(k)),ones(prod(nImgc),1)));
    
    % layer = doubleLayer(K,K,'Bin1',B,'Bin2',B,'activation1',act,'activation2',@identityActivation);
	nL = getBatchNormLayer([prod(nImgc(1:2)) nf(k) 50],'isWeight',1);
    layer = doubleSymLayer(K,'Bin',B,'activation',act,'nLayer',nL);
    blocks{end+1} = ResNN(layer,nt(k),h(k),'useGPU',useGPU,'precision',precision);
    % RegOps{end+1} = opTimeDer(nTheta(blocks{end}),blocks{end}.nt);
    regD = gpuVar(useGPU,precision,repmat([ones(nTheta(K),1); zeros(size(B,2)+nTheta(nL),1)],nt(k),1));
    RegOps{end+1} = opDiag(regD);
    % Connector block
    if doInstNorm && k<numel(h)
%        blocks{end+1} = NN({instNormLayer(nFeatOut(blocks{end}))});
%	    blocks{end+1} = NN({getBatchNormLayer([prod(nImgc(1:2)) nf(k) 50], 'isWeight',0)});

    end
    
    Bc = gpuVar(useGPU,precision,kron(eye(nf(k+1)),ones(prod(nImgc),1)));
    blocks{end+1} = NN({singleLayer(conv(nImgc,[1,1,nf(k),nf(k+1)]),'activation',actc,'Bin',Bc)},'useGPU',useGPU,'precision',precision);
    RegOps{end+1} = opEye(nTheta(blocks{end}));
    if k<numel(h)
        blocks{end+1} = connector(opPoolMCN([nImgc nf(k+1)],2));
        RegOps{end+1} = opEye(nTheta(blocks{end}));
    end
end

%% Connector block
blocks{end+1} = connector(B'/size(B,1));
% blocks{end+1} = connector(opPoolMCN([nImgc nf(end)],2));
RegOps{end+1} = opEye(nTheta(blocks{end}));

%% Put it all together
net    = MegaNet(blocks);
pLoss = softmaxLoss();

theta  = initTheta(net);
W      = 0.1*vec(randn(10,nFeatOut(net)+1));


% RegOpW = blkdiag(opGrad(nImgc/2,nf(end)*10,ones(2,1)),opEye(10));
RegOpW = blkdiag(opEye(numel(W)));
RegOpW.precision = precision;
RegOpW.useGPU = useGPU;

RegOpTh = blkdiag(RegOps{:});
RegOpTh.precision = precision;
RegOpTh.useGPU = useGPU;

pRegW  = tikhonovReg(RegOpW,1e-1,[],'useGPU',useGPU,'precision',precision);
pRegKb = tikhonovReg(RegOpTh,1e-1,[],'useGPU',useGPU,'precision',precision);

%% Prepare optimization
fctn = dnnBatchObjFctn(net,pRegKb,pLoss,pRegW,Y0,C,'batchSize',50,'useGPU',useGPU,'precision',precision);
fval = dnnBatchObjFctn(net,[],pLoss,[],Ytest,Ctest,'batchSize',50,'useGPU',useGPU,'precision',precision);
% fval = [];
%%
opt = sgd();
% if doInstNorm
%    opt.learningRate = @(epoch) 1e-1/sqrt(2*epoch);
% else
    opt.learningRate = 1e-2;
% end
opt.maxEpochs    = 10;
opt.nesterov     = false;
opt.ADAM         = false;
opt.miniBatch    = 50;
opt.momentum     = 0.9;
opt.out          = 1;

%% run optimization
alphaW = logspace(-1,-3,3);
alphaT = logspace(-1,-3,3);
alphaW = 1e-2;
alphaT = 1e-2;
xi0     = [theta(:); W(:)];
best = [0,0,0];
xOpt = xi0;
%% random hyper parameter search

maxTry = 5;
alphaW = 10.^(-5 + 3*rand(maxTry,1));
alphaT = 10.^(-5 + 3*rand(maxTry,1));
lr     = 10.^(-2 + 2*rand(maxTry,1));
mom    = .7+.3*rand(maxTry,1);
optsd = sd()
optsd.maxIter=10;
optsd.out=2
hisAll = cell(0,1)
for k=1:maxTry
	opt.learningRate = lr(k);
	fctn.pRegTheta.alpha = alphaT(k);
	fctn.pRegW.alpha  = alphaW(k);
	opt.momentum = mom(k);
	[xOpt,His] = solve(optsd,fctn,xi0,fval);
	[~,idx] = min(His.his(:,5));
	hisAll{end+1} =  His;
end
keyboard;


for i=1:numel(alphaW)
    fctn.pRegW.alpha = alphaW(i);
    
    fctn.pRegTheta.alpha = alphaT(i);
    [xOpt,His] = solve(opt,fctn,xOpt,fval);
    opt.learningRate = 1e-3;
    [xOpt,His] = solve(opt,fctn,xOpt,fval);
    opt.learningRate = 1e-4;
    [xOpt,His] = solve(opt,fctn,xOpt,fval);

    [thOpt,WOpt] = split(fctn,gather(xOpt));
    save(sprintf('EResNN_STL10_avg_instNorm_%d_%d_%d-%d-%d_aW_%1.1e_aT_%1.1e.mat',numel(h),doInstNorm,nt,nf,fctn.pRegW.alpha,fctn.pRegTheta.alpha),'thOpt','WOpt','His')
    opt.maxEpochs = 50;
    opt.learningRate = @(epoch) 1e-5/sqrt(epoch);
    if max(His.his(:,end)) > best(3);
        best =  [fctn.pRegW.alpha fctn.pRegTheta.alpha max(His.his(:,end))]
    end
end

