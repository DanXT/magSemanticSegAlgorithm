classdef iResNN < abstractMegaNetElement
    % Semi implicit Residual Neural Network block
    %
    % Y_k+1 = (I + h*K'*K)\(Y_k + h*layer{k}(S*Y_k)))
    
    properties
        layerEXP  % layer for explicit time step
        layerIMP  % layer for impilict time step
        nt        % number of time steps
        h         % step size
        outTimes  
        Q
        useGPU
        precision
    end
    
    methods
        function this = iResNN(layerEXP,layerIMP,nt,h,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            outTimes  = zeros(nt,1); outTimes(end)=1;
            Q = 1.0;
            for k=1:2:length(varargin)     % overwrites default parameter
               eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                layerEXP.useGPU = useGPU;
                layerIMP.useGPU = useGPU;
            end
            if not(isempty(precision))
                layerEXP.precision = precision;
                layerIMP.precision = precision;
            end
            this.layerEXP = layerEXP;
            this.layerIMP = layerIMP;
            
            if nFeatOut(layerEXP)~=nFeatIn(layerIMP)
                error('%s - nFeatOut(layerExp)=%d does not match nFeatIn(layerImp)=%d',...
                    mfilename,nFeatOut(layerEXP),nFeatIn(layerIMP));
            end
            if nFeatIn(layerEXP)~=nFeatOut(layerIMP)
                error('%s - nFeatIn=%d does not match nFeatOut=%d',...
                    mfilename,nFeatIn(layerEXP),nFeatOut(layerIMP));
            end
            this.nt    = nt;
            this.h     = h;
            this.outTimes = outTimes;
            this.Q = Q;
        end
        
        function n = nTheta(this)
            n = this.nt*(nTheta(this.layerEXP) + nTheta(this.layerIMP));
        end
        
        function n = nFeatIn(this)
            n = nFeatIn(this.layerEXP);
        end
        
        function n = nFeatOut(this)
            n = nFeatOut(this.layerIMP);
        end
        
        function n = nDataOut(this)
           if numel(this.Q)==1
               n = nnz(this.outTimes)*nFeatOut(this.layerEXP);
           else
               n = nnz(this.outTimes)*size(this.Q,1);
           end
        end
        
        
        function theta = initTheta(this)
            theta = repmat([vec(initTheta(this.layerEXP)); ...
                            vec(initTheta(this.layerIMP))],this.nt,1);
        end
        function [thS,thK] = split(this,theta)
            theta = reshape(theta,[],this.nt);
            %nLin  = nTheta(this.linearLayer);
            nnLin = nTheta(this.layerEXP);
            thS = theta(1:nnLin,:);
            thK = theta(nnLin+1:end,:);
        end
        
        %% ------- apply forward problems -----------
        function [Ydata,Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatIn(this);
            Y   = reshape(Y0,[],nex);
            if nargout>1;    tmp = cell(this.nt+1,3); tmp{1,1} = Y0; end
            
            theta = reshape(theta,[],this.nt);
            [thetaS,thetaK] = this.split(theta);
            
            Ydata = [];
            for i=1:this.nt
                
                [Z,~,tmp{i,2}] = apply(this.layerEXP,thetaS(:,i),Y);
                tmp{i,3}       =  Y + this.h * Z;
                Y              = implicitTimeStep(this.layerIMP.K,...
                                                  thetaK(:,i),tmp{i,3},this.h);
                if nargout>1, tmp{i+1,1} = Y; end
                if this.outTimes(i)==1
                    Ydata = [Ydata;this.Q*Y];
                end
            end
        end
        
        %% -------- Jacobian matvecs ---------------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            dYdata = [];
            theta = reshape(theta,[],this.nt);
            [thetaS,thetaK] = this.split(theta);

            for i=1:this.nt
                dY = dY + this.h* JYmv(this.layerEXP,dY,thetaS(:,i),...
                                       tmp{i,1},tmp{i,2});
                                   
                dY = iJYmv(this.layerIMP,dY,thetaK(:,i),[],[],this.h);
                
                if this.outTimes(i)==1
                    dYdata = [dYdata; this.Q*dY];
                end
            end
        end
        
        %%
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            
            dYdata = [];
            
            [thS,thK]   = split(this,theta);
            [dthS,dthK] = split(this,dtheta);
            
            
            for i=1:this.nt
                  dY = dY + this.h*Jmv(this.layerEXP,dthS(:,i),...
                                       dY,thS(:,i),tmp{i,1},tmp{i,2});
                                   
                  dY = iJmv(this.layerIMP,dthK(:,i),dY,thK(:,i),tmp{i,3},this.h);
                  if this.outTimes(i)==1
                      dYdata = [dYdata;this.Q*dY];
                  end
            end
        end
        
        %% -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W)
                W = 0;
            elseif not(isscalar(W))
                W     = reshape(W,[],nex);
            end
            [thS,thK]   = split(this,theta);
    
            cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                if  this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                W = iJYTmv(this.layerIMP,W,thK(:,i),[],[],this.h); 
                
                dW = JYTmv(this.layerEXP,W,[],thS(:,i),tmp{i,1},tmp{i,2});
                W  = W + this.h*dW;
            end
        end
        
        %%
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative)
               doDerivative =[1;0]; 
            end
            
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W) 
                W = 0;
            elseif numel(W)>1
                W     = reshape(W,[],nex);
            end
            
            [thS,thK]   = split(this,theta);
            
            cnt = nnz(this.outTimes);
            dthK = 0*thK; dthS = 0*thS;
            for i=this.nt:-1:1
                if  this.outTimes(i)==1
                    W = W + this.Q'* squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                [dthk,W]  = iJTmv(this.layerIMP,W,thK(:,i),tmp{i,3},this.h);
                [dmbi,dW] = JTmv(this.layerEXP,W,[],thS(:,i),tmp{i,1},tmp{i,2});
                
                dthS(:,i)  = this.h*dmbi;
                dthK(:,i)  = dthk(:);
                
                W = W + this.h*dW;
            end
            dtheta = vec([dthS; dthK]);
            if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
            end
        end
        
        %% ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.layerEXP.useGPU  = value;
                this.layerIMP.useGPU  = value;
            end
            this.Q = gpuVar(value,this.precision,this.Q);
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layerEXP.precision = value;
                this.layerIMP.precision = value;
            end
            this.Q = gpuVar(this.useGPU,value,this.Q);
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layerEXP.useGPU;
        end
        function precision = get.precision(this)
            precision = this.layerEXP.precision;
        end

        
        function runMinimalExample(~)
            nex = 10;
            nK  = [4 4];
            
            K   = dense(nK);
            D   = linearNegLayer(K);
            S   = doubleSymLayer(K);
            nt  = 10;
            outTimes = zeros(nt,1);
            outTimes([1;10;nt])= 1;
            net = iResNN(S,D,nt,.1,'outTimes',outTimes);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Ydata,~,dA]   = net.apply(mb,Y0);
            dmb = reshape(randn(size(mb)),[],net.nt);
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,dA);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],mb,Y0,dA);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

