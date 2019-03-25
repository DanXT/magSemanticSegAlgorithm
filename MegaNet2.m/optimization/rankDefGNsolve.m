function[theta] = rankDefGNsolve(net,misfit,thW,param)
%[theta] = stochGNsolve(net,misfit,thW,param)
% Solve the DNN problem of
% min misfit(W*YN(th),C) + alpha*R(W,th)
%     YN = net(Y0)
% 

nex   = size(param.Y0,2); 
nth   = nTheta(net);
theta = thW(1:nth);
W     = thW(nth+1:end);

for iter=1:param.maxIter
    
    % get the batch
    batch = randi(nex,param.batchSize,1);
    Y0 = param.Y0(:,batch); C = param.C(:,batch);
    
    % Propagate
    [YN,~,tmp] = apply(net,th,Y);
    S          = W*YN;
    
    [F,dF,d2F] = misfit(S,C);
    
    % Compute a step
    [J,JT] = stochasticJacEstimator(net,th,Y0,tmp);
    
    % approx grad
    dF = JT(dF);
    
    % 