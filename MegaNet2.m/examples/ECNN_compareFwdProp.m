% saves output so we can debug Julia code
clear
[Y,C] = setupMNIST(64);

%%
% build network
nImg = [28 28];
nc   = [1 8 16 32];
nt   = [4;4;4];
h    = [.1;.1;.1];

% opening layer
blocks = cell(0,1);

convKernel = @convFFT;
% convKernel = @convMCN;
K1 = convKernel(nImg,[3 3 nc(1) nc(2)]);
blocks{end+1} = NN({singleLayer(K1)});

%%
for k=1:length(nt)
    % ResNN layers
    K2 = convKernel(nImg,[3 3 nc(k+1) nc(k+1)]);
    
    nL = getTVNormLayer([prod(nImg);nc(k+1)],'isWeight',0);
    L2 = doubleSymLayer(K2,'nLayer',nL);
    RN  = ResNN(L2,nt(k),h(k));
    blocks{end+1} = RN;
    
    if k<length(nt)
        % change channels
        Kc     = convKernel(nImg,[1,1,nc(k+1),nc(k+2)]);
        blocks{end+1} = NN({singleLayer(Kc)});
        
        % change resolution
        Kp = opPoolMCN([nImg nc(k+2)],2);
        blocks{end+1} = connector(Kp);
        nImg = nImg./2;
    end
end
%%
net = MegaNet(blocks);
%%
th = randn(nTheta(net),1);
tic; 
for k=1:10
[Zd,Z,tmp]  = apply(net,th,Y);
end
timeApply = toc/10;

save '/Users/lruthot/Dropbox/DynamicCNN/Meganet2.jl/examples/ECNN_compareFwd.mat' Y Z Zd tmp th timeApply  


