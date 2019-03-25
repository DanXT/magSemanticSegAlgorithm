classdef gradientFiltering < optimizer
    % classdef gradientFiltering < optimizer
    %
    % stochastic gradient descent optimizer for minimizing nonlinear objectives
    
    properties
        maxEpochs
        miniBatch
        numSample
        sigSample
        atol
        rtol
        maxStep
        out
        learningRate
        LH
        P
    end
    
    methods
        
        function this = gradientFiltering(varargin)
            this.maxEpochs = 10;
            this.miniBatch = 16;
            this.numSample = 3;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.learningRate = 0.1;
            this.P = @(x) x;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'epoch', 'Jc','|x-xOld|','learningRate'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e'};
        end
        
        %function [xc,His,xOpt] = solve(this,fctn,xc,fval)
        function[xc] = solve(this,net,loss,xc,Y,C,Yv,CV)
            
            K      = this.numSample;
            lr     = this.learningRate;
            n      = nTheta(net);
            theta  = xc(1:n);
            W      = reshape(xc(n+1:end),size(C,1),[]);

            for i=1:this.maxEpochs
                % Function value and validation
                YV  = apply(net,theta,Yv);
                WYV = W*[YV;ones(1,size(YV,2))];
                [~,tmpV,~] = getMisfitS(loss,WYV,CV);
                
                YN  = apply(net,theta,Y);
                WY  = W*[YN;ones(1,size(YN,2))];
                [F,tmpT,dF,~] = getMisfitS(loss,WY,C);
                
                fprintf('%d.0   %3.3e   %3.3e   %3.3e\n',...
                          i,F,tmpT(3)/tmpT(2),tmpV(3)/tmpV(2))
                
                % compute a descent direction for theta
                Theta  = this.sigSample*randn(n,K);
    
                R  = zeros(numel(YN),K);
                for j=1:K
                    R(:,j) = vec(apply(net,theta+Theta(:,j),Y));
                end
                Rh  = 1/sqrt(K) * (R - mean(R,2)); %(R - YN(:));%
                Th  = 1/sqrt(K) * (Theta - mean(Theta,2));
                dFY = W'*reshape(dF,size(W,1),[]);
                dFY = dFY(1:end-1,:);
                s  = Th*(Rh'*dFY(:));
%                H  = @(x) Th*(Rh'*d2F((Rh*(Th'*x))));
%                [s,~]  = pcg(H,s,1e-2,20);
                %if max(abs(s)) > 1 
                    s = s/max(abs(s)); 
                %end
                %theta  = proj(this,theta - lr*s);
                mu = lr(i);
                for cnt = 1:3
                    thetaTry  = proj(this,theta - mu*s);
                    Ytry      = apply(net,thetaTry,Y);
                    WYtry     = W*[Ytry;ones(1,size(YN,2))];
                
                    Ftry      = getMisfitS(loss,WYtry,C);
                    fprintf('%d.%d   %3.3e\n',i,cnt,Ftry);
                    if Ftry < F
                        break;
                    end
                    mu = mu/2;
                    if cnt == 8, disp('LSB'); end;
                end
                theta = thetaTry;

                % Descent on W
                W  = W - lr(i)*reshape(dF,size(C,1),[])*[YN;ones(1,size(YN,2))]';

            end
        end

        %%
        function[xc] = solveF(this,net,loss,xc,Y,C,Yv,CV)
            
            K      = this.numSample;
            lr     = this.learningRate;
            n      = nTheta(net);
            theta  = xc(1:n);
            W      = reshape(xc(n+1:end),size(C,1),[]);

            for i=1:this.maxEpochs
                % Function value and validation
                YV  = apply(net,theta,Yv);
                WYV = W*[YV;ones(1,size(YV,2))];
                [~,tmpV,~] = getMisfitS(loss,WYV,CV);
                
                YN  = apply(net,theta,Y);
                WY  = W*[YN;ones(1,size(YN,2))];
                [F,tmpT,dF,d2F] = getMisfitS(loss,WY,C);
                
                fprintf('%d.0   %3.3e   %3.3e   %3.3e\n',...
                          i,F,tmpT(3)/tmpT(2),tmpV(3)/tmpV(2))
                
                [JB,JQ] = netJac(net,theta,Y,K);      
                dFY = W'*reshape(dF,size(W,1),[]);
                dFY = dFY(1:end-1,:);
                s  = JQ*(JB'*dFY(:));
                
                J    = @(x) JB*(JQ'*x);
                JT   = @(x) JQ*(JB'*x);
                WTmv = @(x) vec(W(:,1:end-1)*reshape(x,size(W,2)-1,[]));
                Wmv  = @(x) vec(W(:,1:end-1)'*reshape(x,size(W,1),[]));
                
                H  = @(x) JT(Wmv(d2F*WTmv(J(x)))); 
                [s,~]  = pcg(H,s,1e-2,5);
                if max(abs(s)) > 1 
                  s = s/max(abs(s)); 
                end
                %theta  = proj(this,theta - lr*s);
                mu = 1;
                for cnt = 1:8
                    thetaTry  = proj(this,theta - mu*s);
                    Ytry      = apply(net,thetaTry,Y);
                    WYtry     = W*[Ytry;ones(1,size(YN,2))];
                
                    Ftry      = getMisfitS(loss,WYtry,C);
                    fprintf('%d.%d   %3.3e\n',i,cnt,Ftry);
                    if Ftry < F
                        break;
                    end
                    mu = mu/2;
                    if cnt == 8, disp('LSB'); end;
                end
                theta = thetaTry;

                % Descent on W
                W  = W - lr(i)*reshape(dF,size(C,1),[])*[YN;ones(1,size(YN,2))]';

            end
        end

        
        %%
        
        
        function x = proj(this,x)

            x(x<this.LH(1)) = this.LH(1);
            x(x>this.LH(2)) = this.LH(2);

        end     

    end
end
