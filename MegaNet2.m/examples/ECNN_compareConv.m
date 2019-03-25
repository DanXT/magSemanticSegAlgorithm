% saves output so we can debug Julia code
clear;
[Y,C] = setupMNIST(16);
Y = Y';
Y = [Y; sin(Y); cos(Y)];
%%
% build network
nImg = [28 28];
nc   = [3 4];
K1 = convFFT(nImg,[3 3 nc(1) nc(2)]);
theta = randn(nTheta(K1),1);
% theta = zeros(3,3);
% theta(2,1)=-1;
% theta(2,3)=1;
% theta = theta';
theta = theta(:);
tic; Kop = getOp(K1,theta); toc;
tic; Z = Kop*Y;toc;

% E = eye(size(Kop,1));
% Kmat = Kop*E;
% Kmat = sparse(Kmat);

save /Users/lruthot/Dropbox/DynamicCNN/Meganet2.jl/examples/ECNN_compareConv.mat Y Z theta


