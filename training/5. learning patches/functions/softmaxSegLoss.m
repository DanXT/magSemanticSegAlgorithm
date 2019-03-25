classdef softmaxSegLoss
    % classdef softmaxLoss
    %
    % object describing softmax loss function
    
    properties
       shift
       theta
       addBias
       nClass
       addWeights
    end
   
    
    methods
        function this = softmaxSegLoss(varargin)
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end


            this.shift   = 0;
            this.theta   = 1e-3;
            this.addBias = 1;
            this.nClass  = 1;
            this.addWeights = 0;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        
        
        function [F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,W,Y,C,varargin)
            doDY = (nargout>3);
            doDW = (nargout>1);
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            dWF = []; d2WF = []; dYF =[]; d2YF = [];
            
            
            szY  = size(Y);
            nex  = szY(2);
           
            %%
            nImg = size(C,1)/this.nClass;
            Y = reshape(Y, nImg,[],nex);
            Y = permute(Y,[2 1 3]);
            if this.addBias==1
                Y = cat(1, Y, ones(1,nImg,nex));
            end
            szW  = [this.nClass, size(Y,1)];
            W    = reshape(W,szW);
            Y   = Y - this.shift;
            % change from here!
            Y = reshape(Y,size(Y,1),[]);

            %%
            WY = W*Y;
%             % make sure that the largest number in every row is 0
            m = max(WY,[],1);
            WY = WY - m;
            %%
            
            if this.addWeights == 1
                C_reshape = reshape(C,this.nClass, nImg, nex);
                ratios = sum(C_reshape,2);
                ratios = ratios./sum(ratios,1);
                wts_inv = ratios.*C_reshape;
                wts=1./wts_inv;
                wts(isinf(wts))=0;
                wts = reshape(wts, this.nClass,[]);
            end
            
            
            S    = exp(WY);
            Cp   = getLabels(this,S);
            Ctry = reshape(C, this.nClass,[]);
            
            if this.addWeights == 0
                err  = nnz(Ctry-Cp)/2;
                F    = -sum(sum(Ctry.*(WY))) + sum(log(sum(S,1)));   
            else
                if this.addWeights == 1
                    [Cp1,~] = find(Cp==1);
                    [Ctry1,~] = find(Ctry==1);
                    conf = confusionmat(Ctry1,Cp1);
                    err  = nnz(Ctry-Cp)/2;
                    F    = -sum(sum(wts.*Ctry.*(WY))) + sum(wts.*Ctry,1)*(log(sum(S,1)))';
                end
                
                
            end
            
            para = [F,nex*nImg,err];
            F    = F/nex;
            %%

            if (doDW) && (nargout>=2)
                if this.addWeights == 0
                    dF   = -Ctry + S./sum(S,1); %S./sum(S,2));
                    dWF  = vec(dF*(Y'/nex));
                else
                    if this.addWeights == 1
                        dF = -wts.*Ctry + S.*sum(wts.*Ctry,1)./sum(S,1);
                        dWF = vec(dF*(Y'/nex));
                    end
                    
                end
               
            end
            if (doDW) && (nargout>=3)
                d2F = @(U) this.theta *U + (U.*S)./sum(S,1) - ...
                    S.*(repmat(sum(S.*U,1)./sum(S,1).^2,size(S,1),1));
                matW  = @(W) reshape(W,szW);
                d2WFmv  = @(U) vec((d2F(matW(U/nex)*Y)*Y'));
                d2WF = LinearOperator(prod(szW),prod(szW),d2WFmv,d2WFmv);
            end
            if doDY && (nargout>=4)
                if this.addBias==1
                    W = W(:,1:end-1);
                end
                WdF = reshape(W'*dF,[],nImg,nex);
                WdF = permute(WdF, [2,1,3]);
                dYF  =   vec(WdF)/nex;
            end
            if doDY && nargout>=5
                WI     = @(T) W*T;  %kron(W,speye(size(Y,1)));
                WIT    = @(T) W'*T;
                matY   = @(Y) reshape(Y,szY);
                d2YFmv = @(T) vec(WIT(((d2F(WI(matY(T/nex)))))));
    
                d2YF = LinearOperator(prod(szY),prod(szY),d2YFmv,d2YFmv);
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'F','accuracy'};
            frmt = {'%-12.2e','%-12.2f'};
        end
        function str = hisVals(this,para)
            str = [para(1)/para(2),(1-para(3)/para(2))*100];
        end       
        function Cp = getLabels(this,W,Y)
            if nargin==2
                S = W;
                nex = size(S,2);
            else
                [nf,nex] = size(Y);
                W      = reshape(W,[],nf+1);
                if this.addBias==1
                    Y     = [Y; ones(1,nex)];
                end
                Y     = Y - this.shift;
                
                S      = exp(W*Y);
            end
            P      = S./sum(S,1);
            [~,jj] = max(P,[],1);
            Cp     = zeros(numel(P),1);
            ind    = sub2ind(size(P),jj(:),(1:nex)');
            Cp(ind)= 1;
            Cp     = reshape(Cp,size(P,1),[]);
        end
        function runMinimalExample(~)
            nex = 3;
            numClass = 4;
            nImg = [5 5];
            nCh  = 2;
            W = vec(randn(numClass,nCh+1));
            Y = randn(nImg(1)*nImg(2)*nCh,nex);
            
            C = zeros(numClass, prod(nImg),nex);
            for i = 1:prod(nImg)
                for j = 1:nex
                    cl = randi(numClass);
                    C(cl,i,j) = 1;
                end
            end
            C = reshape(C, numClass*prod(nImg),nex);
            pLoss = softmaxSegLoss('nClass',numClass, 'addWeights', 1);
            
            
            
            %[F,para,dWF,d2WF,dYF,d2YF] = getMisfit(this,W,Y,C,varargin)
            %E = getMisfit(pLoss,W,Y,C);
            %% test W
            [E,~,dE,d2E,~,~] = getMisfit(pLoss,W,Y,C);

            h = 1;
            rho = zeros(20,2);
            dW = randn(size(W));
            for i=1:20
                E1 = getMisfit(pLoss,W+h*dW,Y,C);
                t  = abs(E1-E);
                t1 = abs(E1-E-h*dE(:)'*dW(:));
                %t2 = abs(E1-E-h*dE(:)'*dW(:) - h^2/2 * dW(:)'*vec(d2E(dW)));
                t2 = 0;
                fprintf('%3.2e   %3.2e   %3.2e\n',t,t1,t2)
    
                rho(i,1) = abs(E1-E);
                rho(i,2) = abs(E1-E-h*dE(:)'*dW(:));
                %rho(i,3) = t2;
                h = h/2;
            end

            rho(2:end,:)./rho(1:end-1,:)
            
            %% test Y
            [E,~,~,~,dYE,d2YE] = getMisfit(pLoss,W,Y,C);

            h = 1;
            rho = zeros(20,2);
            dW = randn(size(Y));
            for i=1:20
                E1 = getMisfit(pLoss,W,Y+h*dW,C);
                t  = abs(E1-E);
                t1 = abs(E1-E-h*dYE(:)'*dW(:));
                %t2 = abs(E1-E-h*dE(:)'*dW(:) - h^2/2 * dW(:)'*vec(d2E(dW)));
                t2 = 0;
                fprintf('%3.2e   %3.2e   %3.2e\n',t,t1,t2)
    
                rho(i,1) = abs(E1-E);
                rho(i,2) = abs(E1-E-h*dYE(:)'*dW(:));
                %rho(i,3) = t2;
                h = h/2;
            end

            rho(2:end,:)./rho(1:end-1,:)
        end
    end
    
end

