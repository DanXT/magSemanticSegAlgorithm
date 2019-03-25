function[JS,Jb,JY] = diResNN(K,S,b,Y,param)
% [JK,Jb,JY] = diResNN(K,S,b,Y,dA,param)
% 

activation = param.activation;

[nex,nf] = size(Y(:,:,end));

nt = param.nt;

JK = []; Jb = []; JS = []; JY = speye(nex*nf);

for i=nt:-1:1
    Ki = K(:,:,i);
    Si = S(:,:,i);
    
    
    [~,dAi] = activation(Si*Y(:,:,i) + b(i));
    
    JS  = [h*JY* sdiag(dAi) * kron(speye(size(Si,2)),Y(:,:,i)),JK];
    Jb  = [h*JY * sdiag(dAi) * ones(nex,1),Jb];
    
    JY  = JY + h*JY*(sdiag(dAi)*kron(Si',speye(nex)));
end

