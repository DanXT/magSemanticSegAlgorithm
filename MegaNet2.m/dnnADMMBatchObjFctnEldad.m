classdef dnnADMMBatchObjFctnEldad < objFctn
    % classdef dnnADMMBatchObjFctnEldad < objFctn
    %
    
    properties
        net         % description of DNN to be trained
        pRegTheta   % regularizer for network parameters
        W           % 
        Y           % input features
        Ytarget           % target for features
        batchSize   % batch size
        batchIds    % indices of batches
        useGPU      % flag for GPU computing
        precision   % flag for precision
    end
    
    methods
        function this = dnnADMMBatchObjFctnEldad(net,pRegTheta,W,Y,Ytarget,varargin)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            batchSize = 10;
            batchIds  = randperm(size(Y,2));
            useGPU    = [];
            precision = [];
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            
            this.net       = net;
            this.pRegTheta = pRegTheta;
            if not(isempty(useGPU))
                this.useGPU = useGPU;
            end
            
            if not(isempty(precision))
                this.precision=precision;
            end
            [Ytarget,Y] = gpuVar(this.useGPU,this.precision,Ytarget,Y);
            this.Ytarget         = Ytarget;
            this.W = W;
            this.Y         = Y;
            this.batchSize = batchSize;
            this.batchIds  = batchIds;
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,theta,idx)
            if not(exist('idx','var')) || isempty(idx)
                Ytarget = this.Ytarget;
                Y = this.Y;
            else
                Y = this.Y(:,idx);
                Ytarget = this.Ytarget(:,idx);
            end
                
            compGrad = nargout>2;
            compHess = nargout>3;
            dJth = 0.0;  Hth = [];  PC = [];
            
            
            nex = size(Y,2);
            nb  = nBatches(this,nex);
            
            this.batchIds  = randperm(size(Y,2));
                        
            % compute loss
            F = 0.0; hisLoss = [];
            for k=nb:-1:1
                idk = this.getBatchIds(k,nex);
                if nb>1
                    Yk  = Y(:,idk);
                    Ytargetk  = Ytarget(:,idk);
                else
                    Yk  = Y;
                    Ytargetk  = Ytarget;
                end
                
                if compGrad
                    [YNk,tmp]                  = apply(this.net,theta,Yk); % forward propagation
                    J = getJthetaOp(this.net,theta,Yk,tmp);
                else
                    [YNk]        = apply(this.net,theta,Yk); % forward propagation
%                     [Fk,hisLk]  = getMisfit(this.pLoss,W,YNk,Ck);
                end
                res = this.W'*YNk - Ytargetk;
                Fk  = .5*sum(vec(res.^2));
                F    = F    + Fk;
%                 hisLoss  = [hisLoss;hisLk];
                if compGrad
                    dthFk = J'*(this.W*res);
                    dJth  = dJth + dthFk;
                end
            end
            F    = F/nex;
            Jc   = F;
            if compGrad
                dJth = dJth/nex;
                if compHess
                    Hthmv = @(x) J'*(this.W*(this.W'*(J*(x/nex))));
                    Hth   = LinearOperator(numel(theta),numel(theta),Hthmv,Hthmv);
                end
            end
            para = struct('F',F,'hisLoss',hisLoss);
            
            
            % evaluate regularizer for DNN weights
            if not(isempty(this.pRegTheta))
                [Rth,hisRth,dRth,d2Rth]      = regularizer(this.pRegTheta,theta);
                Jc = Jc + Rth;
                if compGrad
                    dJth = dJth + dRth;
                end
                if compHess
                    Hth  = Hth + d2Rth;
                end
                para.Rth = Rth;
                para.hisRth = hisRth;
            end
            
            
            
            dJ   = dJth;
            
            if nargout>3
                H  = Hth;
            end
            
%             if nargout>4
%                 PC = getThetaPC(this,d2YF,theta,Yk,tmp);
%                 PC = blkdiag(PCth,getPC(this.pRegW));
%             end
            
        end
        

        
        function nb = nBatches(this,nex)
            if this.batchSize==Inf
                nb = 1;
            else
                nb =  ceil(nex/this.batchSize);
            end
        end
        
        function ids = getBatchIds(this,k,nex)
            if isempty(this.batchIds) || numel(this.batchIds) ~= nex
                fprintf('reshuffle\n')
                this.batchIds = randperm(nex);
            end
            ids = this.batchIds(1+(k-1)*this.batchSize:min(k*this.batchSize,nex));
        end
        
        function [str,frmt] = hisNames(this)
%             [str,frmt] = hisNames(this.pLoss);
            str = {'F'};
            frmt = {'%1.2e'};
            if not(isempty(this.pRegTheta))
                [s,f] = hisNames(this.pRegTheta);
                s{1} = [s{1} '(theta)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
         
        end
        
        function his = hisVals(this,para)
            his = para.F;
            if not(isempty(this.pRegTheta))
                his = [his, hisVals(this.pRegTheta,para.hisRth)];
            end
        end
        
        function str = objName(this)
            str = 'dnnBatchObj';
        end
        % ------- functions for handling GPU computing and precision ----
        function this = set.useGPU(this,value)
            if isempty(value)
                return
            elseif(value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                if not(isempty(this.net)); this.net.useGPU       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.useGPU       = value; end
                if not(isempty(this.pRegW)); this.pRegW.useGPU       = value; end
                
                [this.Y,this.C] = gpuVar(value,this.precision,...
                                                         this.Y,this.C);
            end
        end
        function this = set.precision(this,value)
            if isempty(value)
                return
            elseif not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                if not(isempty(this.net)); this.net.precision       = value; end
                if not(isempty(this.pRegTheta)); this.pRegTheta.precision       = value; end
                if not(isempty(this.pRegW)); this.pRegW.precision       = value; end
                
                [this.Y,this.C] = gpuVar(this.useGPU,value,...
                                                         this.Y,this.C);
            end
        end
        function useGPU = get.useGPU(this)
                useGPU = -ones(3,1);
                
                if not(isempty(this.net)) && not(isempty(this.net.useGPU))
                    useGPU(1) = this.net.useGPU;
                end
                if not(isempty(this.pRegTheta)) && not(isempty(this.pRegTheta.useGPU))
                    useGPU(2) = this.pRegTheta.useGPU;
                end
                
                useGPU = useGPU(useGPU>=0);
                if all(useGPU==1)
                    useGPU = 1;
                elseif all(useGPU==0)
                    useGPU = 0;
                else
                    error('useGPU flag must agree');
                end
        end
        function precision = get.precision(this)
            isSingle    = -ones(3,1);
            isSingle(1) = strcmp(this.net.precision,'single');
            if not(isempty(this.pRegTheta)) && not(isempty(this.pRegTheta.precision))
                isSingle(2) = strcmp(this.pRegTheta.precision,'single');
            end
                isSingle = isSingle(isSingle>=0);
            if all(isSingle==1)
                precision = 'single';
            elseif all(isSingle==0)
                precision = 'double';
            else
                error('precision flag must agree');
            end

        end

        function runMinimalExample(~)
            help(mfilename);
            
        end
    end
end










