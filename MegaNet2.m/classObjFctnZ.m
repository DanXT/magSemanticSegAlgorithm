classdef classObjFctnZ < objFctn
    % classdef classObjFctnZ < objFctn
    %
    % Objective function for classification,i.e., 
    %
    %   J(Z) = loss(h(W*Z), C) + R(Z),
    %
    % where 
    % 
    %   W    - weights of the classifier
    %   h    - hypothesis function
    %   Z    - features
    %   C    - class labels
    %   loss - loss function object
    %   R    - regularizer (object)
    
    properties
        pLoss
        pRegZ
        W
        C
    end
    
    methods
        function this = classObjFctnZ(pLoss,pRegZ,W,C)
            
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            
            this.pLoss  = pLoss;
            this.pRegZ  = pRegZ;
            this.W      = W;
            this.C      = C;
            
        end
        
        function [Jc,para,dJ,H,PC] = eval(this,Z)
            
            Z = reshape(Z,size(this.W,2)-1,[]);
            [Jc,hisLoss,~,~,dJ,H] = getMisfit(this.pLoss,this.W,Z,this.C);
            para = struct('F',Jc,'hisLoss',hisLoss);
            
            
            if not(isempty(this.pRegZ))
                [Rc,hisReg,dR,d2R] = regularizer(this.pRegZ,Z);
                para.hisRZ = hisReg;
                para.Rc     = Rc;
                Jc = Jc + Rc; 
                dJ = dJ + vec(dR);
                H = H + d2R;
                para.hisRW = hisReg;
            end

            if nargout>4
                PC = getPC(this.pRegZ);
            end
        end
        
        function [str,frmt] = hisNames(this)
            [str,frmt] = hisNames(this.pLoss);
            if not(isempty(this.pRegZ))
                [s,f] = hisNames(this.pRegZ);
                s{1} = [s{1} '(W)'];
                str  = [str, s{:}];
                frmt = [frmt, f{:}];
            end
        end
        
        function his = hisVals(this,para)
            his = hisVals(this.pLoss,para.hisLoss);
            if not(isempty(this.pRegZ))
                his = [his, hisVals(this.pRegZ,para.hisRZ)];
            end
        end
        
        
        function str = objName(this)
            str = 'classObjFunZ';
        end
        
        function runMinimalExample(~)
            
            pClass = regressionLoss();
            
            nex = 400;
            nf  = 2;
            nc  = 2;
            
            Y = randn(nf,nex);
            C = zeros(nf,nex);
            
            C(1,Y(2,:)>0) = 1;
            C(2,Y(2,:)<=0) = 1;
            
            W = vec(randn(nc,nf+1));
            pReg   = tikhonovReg(speye(numel(W)));
            
            fctn = classObjFctn(pClass,pReg,Y,C);
            opt1  = newton('out',1,'maxIter',20);
            
%             checkDerivative(fctn,W);
            [Wopt,his] = solve(opt1,fctn,W);
        end
    end
end










