function [X,dX] = doubleHamNNsimple(layer1,layer2,dtheta,theta,X)

if nargin==0
    layer  = doubleSymLayer(dense([2,2]));
    theta  = randn(8,1);        
    dtheta = [randn(4,1);randn(4,1)];
    X0      = randn(4,7);
    [X,dX] = doubleHamNNsimple(layer,layer,dtheta,theta,X0);
    for k=1:10
        h = 2^(-k);
        Xt = doubleHamNNsimple(layer,layer,0*dtheta,theta+h*dtheta,X0);
        fprintf('h=%1.2e  err1=%1.2e  err2=%1.2e \n',h,norm(X-Xt),norm(X+h*dX-Xt));
    end
    return
end

X = reshape(X,4,[]);
Y = X(1:2,:);
Z = X(3:4,:);

tmp = cell(1,4); tmp{1,1} = Y; 

th1 = theta(1:4);
th2 = theta(5:8);

[dZ,~,tmp{1,3}] = apply(layer1,th1,Y);
Z = Z - dZ;
tmp{1,2}=Z;
[dY,~,tmp{1,4}] = apply(layer2,th2,Z);
Y = Y + dY;
X = [Y;Z];

if nargin==1; return; end;
% derivative
dZ = 0; dY = 0;
dth1 = dtheta(1:4);
dth2 = dtheta(5:8);
dZ = dZ - Jmv(layer1,dth1,dY,th1,tmp{1,1},tmp{1,3});
dY = dY + Jmv(layer2,dth2,dZ,th2,tmp{1,2},tmp{1,4});
dX = [dY;dZ];
