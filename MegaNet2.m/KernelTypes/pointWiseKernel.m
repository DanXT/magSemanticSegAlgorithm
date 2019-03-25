classdef pointWiseKernel
    % classdef pointWise < handle
    % 
    % linear transformation given by dense matrix
    %
    %   Y(theta,Y0)  = reshape(theta,nK) * Y0 
    %
    
    properties
        nK  % size of the target [nx ny nf] 
        useGPU
        precision
    end
    
    methods
        function this = pointWiseKernel(nK,varargin)
           this.nK = nK;
           useGPU = 0;
           precision = 'double';
            for k=1:2:length(varargin)     % overwrites default parameter
                eval([ varargin{k},'=varargin{',int2str(k+1),'};']);
            end
            this.useGPU = useGPU;
            this.precision = precision;
            
        end
        function this = gpuVar(this,useGPU,precision)
        end
        function this = set.useGPU(this,value)
            if (value~=0) && (value~=1)
                error('useGPU must be 0 or 1.')
            else
                this.useGPU = value;
            end
        end
        
        function this = set.precision(this,value)
            if not(strcmp(value,'single') || strcmp(value,'double'))
                error('precision must be single or double.')
            else
                this.precision = value;
            end
        end
        
        function n = nTheta(this)
            n = prod(this.nK(3)^2);
        end
        
        function n = nFeatIn(this)
            n = prod(this.nK);
        end
        
        function n = nFeatOut(this)
            n = prod(this.nK);
        end
        
        function theta = initTheta(this)
            theta = rand(nTheta(this),1);
        end
            
        function A = getOp(this,theta)
            Af = reshape(theta,this.nK(3),this.nK(3));
            A  = kron(Af, speye(this.nK(1)*this.nK(2)));
        end
        
       function dY = Jthetamv(this,dtheta,~,Y,~)
            % Jacobian matvec.
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY = getOp(this,dtheta)*Y;
       end
       
       function dtheta = JthetaTmv(this,Z,~,Y,~)
            % Jacobian transpose matvec.
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            Z      = reshape(Z,[],nex);
            dtheta   = (Y*Z')';
        end

    end
end

