classdef convCircFFT < convKernel 
    % classdef convBlkDiagFFT < convKernel
    %
    % 2D convolutions applied to all channels assuming circular coupling
    % Computed using FFTs
    %
    % Transforms feature using affine linear mapping
    %
    %     Y(theta,Y0) =  K(theta_1) * Y0 
    %
    %  where 
    % 
    %      K - convolution matrix (computed using FFTs for periodic bc)
    
    properties
         
    end
    
    methods
        function this = convCircFFT(varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end   
        end

        
        function Y = Amv(this,theta,Y)
            nex   = numel(Y)/prod(nImgIn(this));
            nImg = nImgIn(this); 
            if nImg(3) ~= this.sK(3)
                error('Size problem in circular conv')
            end
            % compute convolution
            theta    = reshape(theta,this.sK);
            Theta    = zeros(nImg(1),nImg(2),this.sK(3));
            Theta(1:this.sK(1),1:this.sK(2),1:this.sK(3)) = theta;
            
            Y  = reshape(Y,nImg(1),nImg(2),this.sK(3),nex);

            Yh     = fft_3D(Y);
            Thetah = fft_3D(Theta);

            T  = Thetah .* Yh;
            Y = real(ifft_3D(T));
            Y  = reshape(Y,[],nex);
        end
        
        
        function Y = ATmv(this,theta,Y)
            nex   = numel(Y)/prod(nImgIn(this));
            nImg = nImgIn(this); 
            if nImg(3) ~= this.sK(3)
                error('Size problem in circular conv')
            end
            % compute convolution
            theta    = reshape(theta,this.sK);
            Theta    = zeros(nImg(1),nImg(2),this.sK(3));
            Theta(1:this.sK(1),1:this.sK(2),1:this.sK(3)) = theta;
            
            Y  = reshape(Y,nImg(1),nImg(2),this.sK(3),nex);

            Yh     = fft_3D(Y);
            Thetah = fft_3D(Theta);

            T  = conj(Thetah) .* Yh;
            Y = real(ifft_3D(T));
            Y  = reshape(Y,[],nex);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY     = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y,~)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta    
            nex    =  numel(Y)/nFeatIn(this);
            nImg   = nImgIn(this);
            % K(theta)*Y = F'*(diag(F*Y)*(F*Q*theta))
            % JTZ = Q'*F'*diag(conj(F*Y))*F*Z
            
            Y  = reshape(Y,nImg(1),nImg(2),this.sK(3),nex);
            Z  = reshape(Z,nImg(1),nImg(2),this.sK(3),nex);
            
            Yh = fft_3D(Y);
            Zh = fft_3D(Z);
            T  = ifft_3D(conj(Yh).*Zh);
            dtheta = T(1:this.sK(1),1:this.sK(2),1:this.sK(3),:);
            dtheta = sum(dtheta,4);
            dtheta = real(dtheta(:));
        end
        
        function Z = implicitTimeStep(this,theta,Y,h)
           % A = F'*diag(F*theta)*F
           % inv(h*ATA + I) = F'*diag(1./(1+h*abs(F*theta)^2))*F
           %
            nex   = numel(Y)/prod(nImgIn(this));
            nImg = nImgIn(this); 
            if nImg(3) ~= this.sK(3)
                error('Size problem in circular conv')
            end
            % compute convolution
            theta    = reshape(theta,this.sK);
            Theta    = zeros(nImg(1),nImg(2),this.sK(3));
            Theta(1:this.sK(1),1:this.sK(2),1:this.sK(3)) = theta;
            
            Y  = reshape(Y,nImg(1),nImg(2),this.sK(3),nex);

            Yh     = fft_3D(Y);
            Thetah = fft_3D(Theta);

            T  = Yh ./(h*abs(Thetah).^2+1);
            Z = real(ifft_3D(T));
            Z  = reshape(Z,[],nex);
        end

        
        
        function n = nImgOut(this)
           n = nImgIn(this);
        end
 
        function theta = initTheta(this)
           theta = randn(this.sK);
        end
        
        function runMinimalExample(this)
            
            Y = randn(8,8,6,5);
            K = rand(3,3,6);     
            K(1,1) = 0;
            K(1,1) = -sum(K(:));
            Kc = convCircFFT([8,8,6],[3,3,6]);
            
            tic
            Z = Amv(Kc,K,Y);
            fprintf('Circ conv time = %3.2e\n',toc)
            
            W = randn(size(Z));
            tic
            S = ATmv(Kc,K,W);
            fprintf('CircTrans conv time = %3.2e\n',toc)
            fprintf('Test Av and ATv %3.2e    %3.2e\n', W(:)'*Z(:),S(:)'*Y(:))
            
            dK = randn(size(K));
            Z = Jthetamv(Kc,dK,[],Y,[]); 
            W = randn(size(Z));
            tic
            dS = JthetaTmv(Kc,W,[],Y,[]);
            fprintf('Circ conv TMV time = %3.2e\n',toc)
            fprintf('Test Jv and JTv %3.2e    %3.2e\n', W(:)'*Z(:),dS(:)'*dK(:))
                
            close all
            E = speye(8^2*6);
            A = Amv(Kc,K,full(E));
            A(abs(A)<1e-3)=0;
            subplot(1,2,1)
            spy(A)
            title('Sparsity structure');
            subplot(1,2,2)
            plot(eig(A),'.')
            title('Eigenvalues');
            fprintf('max eig = %3.2e  min eig = %3.2e\n', ...
                     max(real(eig(A))),min(real(eig(A)))) 
        end

    end
end


