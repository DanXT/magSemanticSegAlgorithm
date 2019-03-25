classdef ResNNab3 < abstractMegaNetElement
    % Residual Neural Network with Adam Bashforth Time Stepping
    %
    %
    %   Y_k+1 = Y_k-1 + h/12* [23*L(y_k,theta_k) - 16*L(y_k-1,theta_k-1) + 5 *L(y_k-2,theta_k-2)]
    %
    %  Here, L(Y,theta)=layer(trafo(theta),Y) andthe initial time points
    %  are computed using forward Euler
    %
    %   Y_1   = Y_0
    %   Y_2   = y_1   +   h*L(Y_0,theta_0)
    %   Y_3   = y_2   +   h*L(Y_2,theta_2)
    %
    %  We assume here that the time points for the states are spaced
    %  equidistantly at nt points with step size h, however, the controls theta may be
    %  spaced non-uniformly on the 1D grid ttheta
    
    properties
        layer        % model for the layer (i.e., the nonlinearity)
        ttheta       % time points for controls
        nt           % number of time points for states
        h            % step size for integration
        outTimes     
        Q
        useGPU
        precision
    end
    
    methods
        function this = ResNNab3(layer,nt,h,ttheta,varargin)
            if nargin==0
                this.runMinimalExample;
                return;
            end
            useGPU = [];
            precision = [];
            outTimes = zeros(nt,1); outTimes(end) = 1;
            Q = 1;
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
            this.nt = nt;
            this.h = h;
            if not(exist('ttheta','var')) || isempty(ttheta)
                ttheta = 0:h:nt*h;
            end
            if (ttheta(end)-nt*h) > 1e-10
                error('time points for states and controls must match')
            end
            this.ttheta = ttheta;
            this.outTimes = outTimes;
            this.Q = 1;
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
        function n = nDataOut(this)
            if numel(this.Q)==1
                n = nnz(this.outTimes)*nFeatOut(this.layer);
            else
                n = nnz(this.outTimes)*size(this.Q,1);
            end
        end
        
        function theta = initTheta(this)
            theta = [];
            for k=1:numel(this.ttheta)
                theta = [theta; vec(initTheta(this.layer))];
            end
        end
        
        % ------- apply forward problems -----------
        function [Ydata,Y,tmp] = apply(this,theta,Y0)
            nex = numel(Y0)/nFeatIn(this);
            Y   = reshape(Y0,[],nex);
            if nargout>1;    tmp = cell(this.nt+1,2); tmp{1,1} = Y0; end
            
            theta = reshape(theta,[],numel(this.ttheta));
            Ydata  = [];
            % first forward Euler step
            ti     = 0;
            thetai = inter1D(theta,this.ttheta,ti);
            [Zm2,~,tmp{1,2}] = apply(this.layer,thetai,Y);
            Y      = Y + this.h*Zm2;
            if this.outTimes(1)==1, Ydata = [Ydata; this.Q*Y]; end
            if nargout>1, tmp{2,1} = Y; end
            
            % second forward Euler step
            ti     = this.h;
            thetai = inter1D(theta,this.ttheta,ti);
            [Zm1,~,tmp{2,2}] = apply(this.layer,thetai,Y);
            Y      = Y + this.h*Zm1;
            if this.outTimes(2)==1, Ydata = [Ydata; this.Q*Y]; end
            if nargout>1, tmp{3,1} = Y; end
            
            % now the ab3 steps
            for i=3:this.nt
                ti = (i-1)*this.h;
                
                thetai = inter1D(theta,this.ttheta,ti);
                
                [Z,~,tmp{i,2}] = apply(this.layer,thetai,Y);
                Y =  Y + (this.h/12)*(23*Z - 16*Zm1 + 5*Zm2);
                Zm2 = Zm1;
                Zm1 = Z;
                if this.outTimes(i)==1, Ydata = [Ydata; this.Q*Y]; end
                if nargout>1, tmp{i+1,1} = Y; end
            end
        end
        
        % -------- Jacobian matvecs ---------------
        function [dYdata,dY] = JYmv(this,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            dYdata = [];
            
            % first forward Euler step
            ti = 0;
            thetai = inter1D(theta,this.ttheta,ti);
            dJYm2  = JYmv(this.layer,dY,thetai,tmp{1,1},tmp{1,2});
            dY     = dY + this.h*dJYm2;
            if this.outTimes(1)==1; dYdata = [dYdata;this.Q*dY]; end
            % second forward Euler step
            ti     = this.h;
            thetai = inter1D(theta,this.ttheta,ti);
            dJYm1  = JYmv(this.layer,dY,thetai,tmp{2,1},tmp{2,2});
            dY     = dY + this.h*dJYm1;
            if this.outTimes(2)==1; dYdata = [dYdata;this.Q*dY]; end
            
            % now the ab3 steps
            for i=3:this.nt
                ti = (i-1)*this.h;
                thetai = inter1D(theta,this.ttheta,ti);
                dJY = JYmv(this.layer,dY,thetai,tmp{i,1},tmp{i,2});
                
                dY =  dY + (this.h/12)*(23*dJY - 16*dJYm1 + 5*dJYm2);
                dJYm2 = dJYm1;
                dJYm1 = dJY;
                if this.outTimes(i)==1; dYdata = [dYdata;this.Q*dY]; end
            
            end
        end
        
        
        function [dYdata,dY] = Jmv(this,dtheta,dY,theta,~,tmp)
            if isempty(dY)
                dY = 0.0;
            elseif numel(dY)>1
                nex = numel(dY)/nFeatIn(this);
                dY   = reshape(dY,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            dtheta = reshape(dtheta,[],numel(this.ttheta));
            dYdata = [];
            % first forward Euler step
            ti = 0;
            thetai  = inter1D(theta,this.ttheta,ti);
            dthetai = inter1D(dtheta,this.ttheta,ti);
            dJm2    = Jmv(this.layer,dthetai,dY,thetai,tmp{1,1},tmp{1,2});
            dY      = dY + this.h*dJm2;
            if this.outTimes(1)==1; dYdata = [dYdata;this.Q*dY]; end
            
            % second forward Euler step
            ti      = this.h;
            thetai  = inter1D(theta,this.ttheta,ti);
            dthetai = inter1D(dtheta,this.ttheta,ti);
            dJm1    = Jmv(this.layer,dthetai,dY,thetai,tmp{2,1},tmp{2,2});
            dY      = dY + this.h*dJm1;
            if this.outTimes(2)==1; dYdata = [dYdata;this.Q*dY]; end
            
            
            for i=3:this.nt
                ti      = (i-1)*this.h;
                thetai  = inter1D(theta,this.ttheta,ti);
                dthetai = inter1D(dtheta,this.ttheta,ti);
                
                dJ    = Jmv(this.layer,dthetai,dY,thetai,tmp{i,1},tmp{i,2});
                
                dY =  dY + (this.h/12)*(23*dJ - 16*dJm1 + 5*dJm2);
                if this.outTimes(i)==1; dYdata = [dYdata;this.Q*dY]; end
            
                dJm2 = dJm1;
                dJm1 = dJ;
            end
        end
        
        % -------- Jacobian' matvecs ----------------
        function alpha = getIntegrationWeights(this,i)
            switch i
                case 2
                    alpha = [this.h -16*this.h/12 5*this.h/12];
                case 1
                    alpha = [this.h 0 5*this.h/12];
                case this.nt
                    alpha = [23*this.h/12 0 0];
                case this.nt-1
                    alpha = [23*this.h/12 -16*this.h/12 0];
                otherwise
                    alpha = [23*this.h/12 -16*this.h/12 5*this.h/12];
            end
        end
        
        function W = JYTmv(this,Wdata,W,theta,Y,tmp)
            nex = numel(Y)/nFeatIn(this);
            if ~isempty(Wdata)
                Wdata = reshape(Wdata,[],nnz(this.outTimes),nex);
            end
            if isempty(W)
                W = 0;
            else
                W     = reshape(W,[],nex);
            end
            theta  = reshape(theta,[],numel(this.ttheta));
            
            Wp1 = 0; Wp2 = 0;  cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                if i==this.nt || this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                alpha  = getIntegrationWeights(this,i);
                ti     = (i-1)*this.h;
                thetai = inter1D(theta,this.ttheta,ti);
                
                temp = alpha(1)*W + alpha(2)*Wp1 + alpha(3)*Wp2;
                
                dW = JYTmv(this.layer,temp,[],thetai,tmp{i,1},tmp{i,2});
                
                Wtt = W;
                W  = W + dW;
                Wp2 = Wp1;
                Wp1 = Wtt;
            end
            
        end
        
        function [dtheta,W] = JTmv(this,Wdata,W,theta,Y,tmp,doDerivative)
            if not(exist('doDerivative','var')) || isempty(doDerivative); 
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
            
            theta  = reshape(theta,[],numel(this.ttheta));
            dtheta = 0*theta;
            
            Wp1 = 0; Wp2 = 0; W = 0.0; cnt = nnz(this.outTimes);
            for i=this.nt:-1:1
                if i==this.nt || this.outTimes(i)==1
                    W = W + this.Q'*squeeze(Wdata(:,cnt,:));
                    cnt = cnt-1;
                end
                
                alpha  = getIntegrationWeights(this,i);
                ti     = (i-1)*this.h;
                [thetai,wi,idi] = inter1D(theta,this.ttheta,ti);
                
                temp = alpha(1)*W + alpha(2)*Wp1 + alpha(3)*Wp2;
                
                [dth,dW] = JTmv(this.layer,temp,[],thetai,tmp{i,1},tmp{i,2});
                
                dtheta(:,idi(1)) = dtheta(:,idi(1)) + wi(1)*dth;
                dtheta(:,idi(2)) = dtheta(:,idi(2)) + wi(2)*dth;
                Wtt = W;
                W  = W + dW;
                Wp2 = Wp1;
                Wp1 = Wtt;

            end
             dtheta = vec(dtheta);
             if nargout==1 && all(doDerivative==1)
                dtheta=[dtheta(:); W(:)];
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
            nt  = 10;
            D   = dense(nK);
            S   = singleLayer(D);
%             outTimes = zeros(nt,1); outTimes(2:2:end) = 1;
            hth = rand(nt,1); hth = hth/sum(hth);
            tth = [0;cumsum(hth)];
            nt  = 10;
            h   = 1/nt;
            
            net = ResNNab3(S,nt,h,tth);
            mb  = randn(nTheta(net),1);
            
            Y0  = randn(nK(2),nex);
            [Ydata,~,tmp]   = net.apply(mb,Y0);
            dmb = reshape(randn(size(mb)),[],numel(net.ttheta));
            dY0  = 0*randn(size(Y0));
            
            dY = net.Jmv(dmb(:),dY0,mb,Y0,tmp);
            for k=1:14
                hh = 2^(-k);
                
                Yt = net.apply(mb+hh*dmb(:),Y0+hh*dY0);
                
                E0 = norm(Yt(:)-Ydata(:));
                E1 = norm(Yt(:)-Ydata(:)-hh*dY(:));
                
                fprintf('h=%1.2e\tE0=%1.2e\tE1=%1.2e\n',hh,E0,E1);
            end
            
            W = randn(size(Ydata));
            t1  = W(:)'*dY(:);
            
            [dWdmb,dWY] = net.JTmv(W,[],mb,Y0,tmp);
            t2 = dmb(:)'*dWdmb(:) + dY0(:)'*dWY(:);
            
            fprintf('adjoint test: t1=%1.2e\tt2=%1.2e\terr=%1.2e\n',t1,t2,abs(t1-t2));
        end
    end
    
end

