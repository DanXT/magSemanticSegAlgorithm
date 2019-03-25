clear
close all
addpath ../../MNIST/

%%

nex = 10;
[Y0,C] = setupMNIST(nex);
Y0 = Y0';
    
%%


nt  = 5;

K = zeros(28^2,28^2,nt);
S = zeros(28^2,28^2,nt);

for i=1:nt
    Ki = full(sprandn(28^2,28^2,0.01));
    Si = -full(diag(rand(28^2,1)));
    %Si = Si-Si';
    K(:,:,i) = Ki;
    S(:,:,i) = Si;   
    b(i)     = randn(1)*0;
end

param.h = 0.01;
param.activation = @tanhActivation;

[Y] = iResNN(K,S,b,Y0,param);

param.nt = nt;

% derivative check
VY    = randn(size(Y0));
[JS,Jb,JY] = diResNN(K,S,b,Y,param);

return
JKb = [JK Jb];
vKb = [vec(vK);vec(vb)];
dYV   = reshape(JY*vec(VY) + JKb*vKb,size(Y0)) ;

for k=1:20
    h  = 2^(-k);
    Kt = K; bt = b;
    for i=1:param.nt
        Kt{i} = K{i} + h*vK{i};
         bt{i} = b{i} + h*vb{i};
    end
    Yk = ResNN(Kt,bt,Y0+h*VY,param);
    
    E0 = norm(Yk-Y);
    E1 = norm(Yk-Y-h*dYV);
    
    fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',h,E0,E1);
end