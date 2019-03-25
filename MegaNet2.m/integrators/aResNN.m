classdef aResNN < abstractMegaNetElement
    % adaptive Residual Neural Network block
    %
    % Y_k+1 = Y_k + h*layer{k}(trafo(theta{k}),Y_k))
    %
    % here adaptive time stepping for Y and theta
    
    properties
        layer
        tY
        ttheta
        useGPU
        precision
    end
    
    methods
        function this = aResNN(layer,tY,ttheta,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
               eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if not(isempty(useGPU))
                layer.useGPU = useGPU;
            end
            if not(isempty(precision))
                layer.precision = precision;
            end
            this.layer = layer;
            if nFeatOut(layer)~=nFeatIn(layer)
                error('%s - dim. of input and output features must agree for ResNet layers',mfilename);
            end
            this.tY     = tY;
            this.ttheta = ttheta;
        end
        
        function n = nTheta(this)
            n = numel(this.ttheta)*nTheta(this.layer);
        end
        function n = nFeatIn(this)
            n = nFeatIn(this.layer);
        end
        function n = nFeatOut(this)
            n = nFeatOut(this.layer);
        end
        
        function theta = initTheta(this)
            theta = [];
            for k=1:numel(this.ttheta)
                theta = [theta; vec(initTheta(this.layer))];
            end
        end

        % ------- apply forward problems -----------
        function [Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatOut(this);
            Y   = reshape(Y0,[],nex);
            if nargout>1;    tmp = cell(numel(this.tY)+1,2); tmp{1,1} = Y0; end
            
            theta = reshape(theta,[],numel(this.ttheta));
            
            for i=1:numel(this.tY)-1
                ti = this.tY(i); hi = this.tY(i+1)-this.tY(i);
                
                thetai = inter1D(theta,this.ttheta,ti);
                
                [Z,tmp{i,2}] = apply(this.layer,thetai,Y);
                Y =  Y + hi * Z;
                if nargout>1, tmp{i+1,1} = Y; end
            end
        end
        
        % -------- Jacobian matvecs ---------------
        function dY = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatOut(this);
                dY   = reshape(dY,[],nex);
            end
            
            theta  = reshape(theta,[],numel(this.ttheta));
            for i=1:numel(this.tY)-1
                ti = this.tY(i); hi = this.tY(i+1)-ti;
                thetai = inter1D(theta,this.ttheta,ti);
                dY = dY + hi* JYmv(this.layer,dY,thetai,tmp{i,1},tmp{i,2});
            end
        end
        
        
        function dY = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatOut(this);
                dY   = reshape(dY,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            dtheta = reshape(dtheta,[],numel(this.ttheta));
            for i=1:numel(this.tY)-1
                  ti      = this.tY(i); hi = this.tY(i+1)-ti;
                  thetai  = inter1D(theta,this.ttheta,ti);
                  dthetai = inter1D(dtheta,this.ttheta,ti);
                  dY = dY + hi* Jmv(this.layer,dthetai,dY,thetai,tmp{i,1},tmp{i,2});
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        
        function W = JYTmv(this,W,theta,~,tmp)
            nex = numel(W)/nFeatOut(this);
            W   = reshape(W,[],nex);
            theta  = reshape(theta,[],numel(this.ttheta));
            for i=numel(this.tY):-1:2
                ti     = this.tY(i); hi = ti-this.tY(i-1);
                thetai = inter1D(theta,this.ttheta,ti);
                dW = JYTmv(this.layer,W,thetai,tmp{i,1},tmp{i,2});
                W  = W + hi*dW;
            end
        end
        
        function [dtheta,W] = JTmv(this,W,theta,~,tmp)
            nex = numel(W)/nFeatOut(this);
            W   = reshape(W,[],nex);
            
            theta  = reshape(theta,[],numel(this.ttheta));
            dtheta = 0*theta;
            for i=numel(this.tY)-1:-1:1
                ti     = this.tY(i); hi = this.tY(i+1)-ti;
                [thetai,wi,idi] = inter1D(theta,this.ttheta,ti);
                [dth,dW] = JTmv(this.layer,W,thetai,tmp{i,1},tmp{i,2});
                dtheta(:,idi(1)) = dtheta(:,idi(1)) + (hi*wi(1))*dth;
                dtheta(:,idi(2)) = dtheta(:,idi(2)) + (hi*wi(2))*dth;
                W = W + hi*dW;
            end
                
        end
        % ------- functions for handling GPU computing and precision ---- 
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.layer.useGPU  = value;
            end
        end
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.layer.precision = value;
            end
        end
        function useGPU = get.useGPU(this)
            useGPU = this.layer.useGPU;
        end
        function precision = get.precision(this)
            precision = this.layer.precision;
        end

        
        function runMinimalExample(~)
            nex = 10;
            nK  = [4 4];
            
            D   = dense(nK);
            S   = singleLayer(D);
            hth = rand(10,1); hth = hth/sum(hth);
            tth = [0;cumsum(hth)];
            hY  = rand(20,1); hY  = hY/sum(hY);
            tY = [0;cumsum(hY)];
            
            net = aResNN(S,tY,tth);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Y,tmp]   = net.apply(mb,Y0);
            dmb = reshape(randn(size(mb)),[],numel(net.ttheta));
            dY0  = randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Y(:));
                E1 = norm(Yt(:)-Y(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Y));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

