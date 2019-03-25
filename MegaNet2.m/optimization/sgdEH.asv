classdef sgdEH < optimizer
    % classdef sgd < optimizer
    %
    % stochastic gradient descent optimizer for minimizing nonlinear objectives
    
    properties
        maxEpochs
        miniBatch
        atol
        rtol
        maxStep
        out
        learningRate
        momentum
        nesterov
		ADAM
    end
    
    methods
        
        function this = sgd(varargin)
            this.maxEpochs = 10;
            this.miniBatch = 16;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.learningRate = 0.1;
            this.momentum  = .9;
            this.nesterov  = true;
			this.ADAM      = false;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if this.ADAM && this.nesterov
                warning('sgd(): ADAM and nestrov together - choosing ADAM');
                this.nesterov  = false;
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'epoch', 'Jc','|x-xOld|','learningRate'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e'};
        end
        
        function xc = solve(this,fctn,xc,fval)
            if not(exist('fval','var')); fval = []; end;
            
            [str,frmt] = hisNames(this);
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt] = parseObjFctn(this,fval);
            
            % evaluate training and validation
            
            epoch = 1;
            dJ = 0*xc;
            mJ = 0;
            vJ = 0;
            if this.ADAM
                mJ = 0*xc;
                vJ = 0*xc;
%                 this.learningRate = 0.001;
            end
            beta2 = 0.999;
            beta1 = this.momentum;
            
            if isnumeric(this.learningRate)
                learningRate    = @(epoch) this.learningRate;
            elseif isa(this.learningRate,'function_handle')
                learningRate  = this.learningRate;
            else
                error('%s - learningRate must be numeric or function',mfilename);
            end
            
            if this.out>0
                fprintf('== sgd (n=%d,maxEpochs=%d,maxStep=%1.1e, lr = %1.1e, momentum = %1.1e, ADAM = %d, Nesterov = %d, miniBatch=%d) ===\n',...
                    numel(xc), this.maxEpochs, this.maxStep,learningRate(1) ,this.momentum, this.ADAM,this.nesterov,this.miniBatch);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
            his = zeros(1,numel(str));
            
            while epoch <= this.maxEpochs
                nex = size(objFctn.Y,2);
                ids = randperm(nex);
                lr = learningRate(epoch);
                for k=1:ceil(nex/this.miniBatch)
                    idk = ids((k-1)*this.miniBatch+1: min(k*this.miniBatch,nex));
                    [Jk,~,dJk] = fctn(xc,idk); 
                    dJ = lr*dJk + this.momentum*dJ;
                    
                    xc = xc - dJ;

                    % Sample to display training and validation error
                    disBatch = fix(this.miniBatch/2);
                    [Jc,tt1] = fctn(xc,randi(nex,disBatch,1));             
                    [Jv,tt2] = fval(xc,randi(200,disBatch,1));
                    
                    fv(k) = Jv; 
                    ft(k) = Jc;
                    erv(k) = tt2.hisLoss(3)/tt2.hisLoss(2);
                    ert(k) = tt1.hisLoss(3)/tt1.hisLoss(2);
                    
                    fprintf('%3.2e  %3.2e   %3.2e   %3.2e\n',...
                             Jc,Jv,...
                             tt1.hisLoss(3)/tt1.hisLoss(2),...
                             tt2.hisLoss(3)/tt2.hisLoss(2))
             
                end
                
                
                epoch = epoch + 1;
                JJ = length(ft)-10:length(ft);
                fprintf('\n\n === Epoch %d  %3.2e   %3.2e   %3.2e   %3.2e ======\n\n',...
                          epoch,mean(ft(JJ)),mean(fv(JJ)),...
                          mean(ert(JJ)),mean(erv(JJ)));
            end
        end
        
        function [fctn,objFctn,objNames,objFrmt,objHis] = parseObjFctn(this,fctn)
            if exist('fctn','var') && not(isempty(fctn)) && isa(fctn,'objFctn')
                objFctn  = fctn;
                [objNames,objFrmt] = objFctn.hisNames();
                objHis   = @(para) objFctn.hisVals(para);
                fctn = @(x,ids) eval(fctn,x,ids);
            else
                objFctn  = [];
                objNames = {};
                objFrmt  = {};
                objHis   = @(x) [];
            end
        end

    end
end