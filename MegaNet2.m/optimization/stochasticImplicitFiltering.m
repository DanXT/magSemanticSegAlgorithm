classdef stochasticImplicitFiltering < optimizer
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
        UpperBound
        LowerBound
        samplingRadius
        
    end
    
    methods
        
        function this = stochasticImplicitFiltering(varargin)
            this.maxEpochs = 10;
            this.miniBatch = 16;
            this.atol    = 1e-3;
            this.rtol    = 1e-3;
            this.maxStep = 1.0;
            this.out     = 0;
            this.UpperBound   = 1e8;
            this.LowerBound   = -1e8;
            this.samplingRadius = 1;
            this.learningRate = 0.1;
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'epoch', 'Jc','|x-xOld|','learningRate'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e'};
        end
        
        function [xc,His,xOpt] = solve(this,fctn,xc,fval)
            if not(exist('fval','var')); fval = []; end;
            
            [str,frmt] = hisNames(this);
            
            % parse objective functions
            [fctn,objFctn,objNames,objFrmt,objHis]     = parseObjFctn(this,fctn);
            str = [str,objNames{:}]; frmt = [frmt,objFrmt{:}];
            [fval,obj2Fctn,obj2Names,obj2Frmt,obj2His] = parseObjFctn(this,fval);
            str = [str,obj2Names{:}]; frmt = [frmt,obj2Frmt{:}];
            doVal     = not(isempty(obj2Fctn));
            optVal    = 0;
            
            % evaluate training and validation
            
            epoch = 1;
            xOld = xc;
            
            if isnumeric(this.learningRate)
                learningRate    = @(epoch) this.learningRate;
            elseif isa(this.learningRate,'function_handle')
                learningRate  = this.learningRate;
            else
                error('%s - learningRate must be numeric or function',mfilename);
            end
            
            if this.out>0
                fprintf('== sgd (n=%d,maxEpochs=%d,maxStep=%1.1e, lr = %1.1e, momentum = %1.1e, ADAM = %d, Nesterov = %d, miniBatch=%d) ===\n',...
                    numel(xc), this.maxEpochs, this.maxStep,learningRate(1) ,this.miniBatch);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
            
            his = zeros(1,numel(str));
            nex = size(objFctn.Y,2);
            ids = randperm(nex);
            % implicit sampling
            [Jc,dJ] = filterObjFun(this,xc); 
            
            while epoch <= this.maxEpochs
                    
                mu = learningRate(epoch);
                lsiter = 1;
                while 1
                    xt = projX(this,xc - mu*dJ);
                    [Jt,dJt] = filterObjFun(this,xc);
                    Jt = median(Jt);
                    if Jt < Jc, break; end
                    mu = mu/2;
                    lsiter = lsiter+1;
                    if lsiter>10, disp('LSB'); end
                end
                xc = xt;
                dJ = dJt;
                % we sample 2^12 images from the training set for displaying the objective.     
                [Jc,para] = fctn(xc,ids(1:min(nex,2^12))); 
                if doVal
                    [Fval,pVal] = fval(xc,[]);
                    valAcc = obj2His(pVal);
                    if valAcc(2)>optVal
                        xOpt = xc;
                        optVal = valAcc(2);
                    end
                end
                his(epoch,1:4)  = [epoch,gather(Jc),gather(norm(xOld(:)-xc(:))),lr];
                if this.out>0
                    fprintf([frmt{1:4}], his(epoch,1:4));
                end
                xOld       = xc;
                
                if size(his,2)>=5
                    his(epoch,5:end) = [gather(objHis(para)), gather(obj2His(pVal))];
                    if this.out>0
                        fprintf([frmt{5:end}],his(epoch,5:end));
                    end
                end
                fprintf('\n');
                epoch = epoch + 1;
            end
            His = struct('str',{str},'frmt',{frmt},'his',his(1:min(epoch,this.maxEpochs),:));
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
        function[x] = projX(this,x)
            x(x>this.UpperBound) = this.UpperBound;
            x(x<this.LowerBound) = this.LowerBound;
        end
        
        function [Jc,dJ] = filterObjFun(this,xc)
            Jc = []; dJc = [];
            for k=1:ceil(nex/this.miniBatch)
                idk = ids((k-1)*this.miniBatch+1: min(k*this.miniBatch,nex));
                delta = randn(size(xc)) * this.samplingRadius;
                [Jk,~,dJk] = fctn(xc+delta,idk); 
                Jc  = [Jc; Jk];
                dJc = [dJc dJk];
            end
            Jc = median(Jc);
            dJ = median(dJc,2);
        end
    end
end