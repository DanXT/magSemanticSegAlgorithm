classdef iPALM < optimizer
    % classdef iPALM < optimizer
    %
    % iPALM optimizer. Thanks to Thomas Pock (Graz) for suggesting this
    % optimizer, sharing a first version of the code, and a lot of helpful
    % suggestions.
    %
    % The algorithm is presented in this work. Please cite when using this
    % solver:
    %
    % @article{PockEtAl2017,
    %   author = {Pock, Thomas and Sabach, Shoham},
    %   title = {{Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems}},
    %   journal = {arXiv.org},
    %   year = {2017},
    %   eprint = {1702.02505v1},
    %   eprintclass = {math.OC},
    %   doi = {10.1137/16M1064064},
    % }


    
    properties
        maxEpochs  
        miniBatch
        out
        ids        % indices of components for each block
        Lip        % Lipschitz constants for each block
        Prox       % proximal operators
        targetLoss 
    end
    
    methods
        
        function this = iPALM(ids,varargin)
            nb             = size(ids,2);
            this.maxEpochs = 10;
            this.miniBatch = 16;
            this.out       = 1;
            this.Lip       = 1e10*ones(nb,1);
            this.Prox      = cell(nb,1); 
            this.targetLoss = 0;
            for k=1:nb; this.Prox{k} = @(x,tau) x; end
            for k=1:2:length(varargin)     % overwrites default parameter
                eval(['this.' varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            if nb~=numel(this.Prox)
                error('number of blocks %d must match number of proximal operators %d',nb,numel(Prox));
            end
            this.Prox = this.Prox;
            this.ids = ids;
        end
        
        function [str,frmt] = hisNames(this)
            str  = {'epoch', 'Jc','|x-xOld|','accel.','ProxSteps','min(L)','max(L)'};
            frmt = {'%-12d','%-12.2e','%-12.2e','%-12.2e','%-12d','%-12.2e','%-12.2e'};
        end
        
        function [xc,His,Lip,xOpt] = solve(this,fctn,xc,fval)
            if not(exist('fval','var')); fval = []; end
            Lip = this.Lip;
            nb = size(this.ids,2);
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
            xOldStep = xc;
            if this.out>0
                fprintf('== iPALM (n=%d,blocks=%d,maxEpochs=%d,range(Lip)=[%1.1e,%1.1e], miniBatch=%d) ===\n',...
                    numel(xc),nb, this.maxEpochs,min(Lip),max(Lip), this.miniBatch);
                fprintf([repmat('%-12s',1,numel(str)) '\n'],str{:});
            end
                
            his = zeros(1,numel(str));
            while epoch <= this.maxEpochs
                nex  = size(objFctn.Y,2);
                idex = randperm(nex);
                 acc   = (epoch-1)/(epoch+2);

                 PS = 0;
                   
                for k=1:ceil(nex/this.miniBatch)
                    % update each block separately
                   idk = idex((k-1)*this.miniBatch+1: min(k*this.miniBatch,nex));
                   for block=1:nb
                       yc    = xc + acc*(xc-xOld).*this.ids(:,block);
                       xOld  = xc;
                       idb   = this.ids(:,block);
                       [J0,~,dJk] = fctn(yc,idk);
                       
                       for j=1:10
                           % proximal step xc = prox_tau_k^Rk(yc-dJk)
                           
                           rhs = yc - (1/Lip(block))*(dJk.*idb);
                           tt = this.Prox{block}(rhs(idb==1),Lip(block));
                           xc(idb==1) = tt;
                           
                           Jt    = fctn(xc,idk);
                           Q     = J0 + sum(vec(dJk.*(xc-yc))) + Lip(block)/2*sum(vec(xc-yc).^2);
                           PS    = PS + 1;
                           if Jt <= Q 
                               Lip(block) = Lip(block)/1.5;
                               break;
                           else
                               Lip(block) = Lip(block)*2;
                           end
                       end
                   end
                end
                % we sample 2^12 images from the training set for displaying the objective.     
                [Jc,para] = fctn(xc,idex(1:min(nex,2^12))); 
                if doVal
                    [Fval,pVal] = fval(xc,[]);
                    valAcc = obj2His(pVal);
                    if (nargout>3) && (valAcc(2)>optVal)
                        xOpt = xc;
                        optVal = valAcc(2);
                    end
                    valHis = gather(obj2His(pVal));
                else
                    valHis =[];
                end
                
                his(epoch,1:7)  = [epoch,gather(Jc),gather(norm(xOldStep(:)-xc(:))),acc,PS,min(Lip),max(Lip)];
                if this.out>0
                    fprintf([frmt{1:4}], his(epoch,1:4));
                end
                xOldStep       = xc;
                
                if size(his,2)>=5
                    his(epoch,8:end) = [gather(objHis(para)), valHis];
                    Lc = gather(objHis(para)); Lc = Lc(1);
                    if this.out>0
                        fprintf([frmt{5:end}],his(epoch,5:end));
                    end
                end
                if this.out>0
                    fprintf('\n');
                end
                if exist('Lc','var')  && Lc<this.targetLoss
                    fprintf('--- %s achieved target loss of %1.2e at iteration %d---\n',mfilename,this.targetLoss,epoch);
                    break
                end
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

    end
end