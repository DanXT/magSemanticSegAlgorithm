function [uEChain, objFun] = EnsembleKFSoftMax(G, C, uE, maxIter)
%[uEChain, objFun] = EnsembleKFSoftMax(G, C, uE, maxIter, mu)
% Solve the KF using the dual
% G forward mapping
% C data
% uE particles
% maxIter 
%

nE      = size(uE, 2);
objFun  = zeros(maxIter + 1, nE,'like',C);

% Propagate forward
GuE            = reshape(G(uE),size(C,1),size(C,2),[]);
[E,dE,d2E]     = softMaxFun(GuE,C);
objFun(1, :)   = E';
uEChain        = zeros([size(uE), maxIter+1],'like',C);
uEChain(:,:,1) = uE;

for iter = 1:maxIter

    fprintf('iter=%d.0\tmin(J)=%1.2e\tmax(J)=%1.2e\n',...
             iter,min(objFun(iter,:)),max(objFun(iter,:)));

    % approximate Jacobians
    uEmean  = mean(uE, 2);
    GuEmean = mean(GuE, 3);
    
    Ju    = 1/sqrt(nE-1)*(uE  - uEmean );
    Jw    = 1/sqrt(nE-1)*(GuE - GuEmean);
    
    s = 1; %svd(Jw,'econ');
    epsilon = max(s)/10;
    % update ensemble
    %   dX  = (Jw'*d2E*Jw + epsilon*I)\(Jw'*dE)
    
    Jw = reshape(Jw,size(Jw,1)*size(Jw,2),[]);
    
    dE = reshape(dE,[],nE);
    tensor = @(t) reshape(t,size(C,1),size(C,2),[]);
    mat    = @(t) reshape(t,[],nE);  
    H      = @(x) Jw'*(mat(d2E(tensor(Jw*x)))) + epsilon*x;
    Id     = @(x) x;
    
    dX  = -Jw'*dE; %blkPCG(H,-Jw'*dE,Id,10,1e-3,1);
    duE = Ju*dX;
    
    % update carefully
    mu = 10; cnt = 1;
    while 1
        uEtry  = uE + mu*duE;
    
        GuE                 = reshape(G(uEtry),size(C,1),size(C,2),[]);
        [Etry,dE,d2E]       = softMaxFun(GuE,C);
        
        fprintf('iter=%d.%d\tmin(J)=%1.2e\tmax(J)=%1.2e\n',...
             iter,cnt,min(Etry),max(Etry));
        
        if min(Etry) < min(E)
            break;
        end
        mu = mu/2;
        cnt = cnt+1;
        if cnt>1
            disp('LSB');break
        end
    end
    E = Etry;
    uE = uEtry;
    
    objFun(iter+1, :)   = E';
    uEChain(:,:,iter+1) = uE;
end
