classdef convFFTSF < convKernel 
    % classdef convFFTSF < convKernel
    % 2D coupled convolutions. Computed using FFTs
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
        function this = convFFTSF(varargin)
            this@convKernel(varargin{:});
            if nargout==0 && nargin==0
                this.runMinimalExample;
                return;
            end
            
        end
        
        function runMinimalExample(~)
            nImg   = [16 16];
            sK     = [3 3,2,4];
            kernel = feval(mfilename,nImg,sK);
            theta1 = rand(sK); 
            theta1(:,1,:) = -1; theta1(:,3,:) = 1;
            theta  = [theta1(:);];

            I  = rand(nImgIn(kernel)); I(4:12,4:12,:) = 2;
            Ik = reshape(Amv(kernel,theta,I),kernel.nImgOut());
            figure(1); clf;
            subplot(1,2,1);
            imagesc(I(:,:,1));
            title('input');
            
            subplot(1,2,2);
            imagesc(Ik(:,:,1));
            title('output');
        end
        
        function Z = Amv(this,theta,Y)
            nex   = numel(Y)/prod(nImgIn(this));
            sK    = this.sK;
            nImg  = this.nImg;
            K     = reshape(theta,sK);
            Kf    = zeros(this.nImg(1),nImg(2),sK(3),sK(4));
            Kf(1:sK(1),1:sK(2),:,:) = K;
            
            % FFT for the kernels
            Kfh = fft2(Kf);
            % FFt on the Data
            Yh  = reshape(Y,nImg(1),nImg(2),sK(4),nex);
            Yh  = fft2(Yh);
            
            %for loop on the product
            Zh  = zeros(nImg(1),nImg(2),sK(3),nex); 
            for i=1:sK(3)
                for j=1:sK(4)
                    Zh(:,:,i,:) = Zh(:,:,i,:) + Kfh(:,:,i,j).*Yh(:,:,j,:);
                end
            end
            Z = ifft2(Zh);
        end        
        function ATY = ATmv(this,theta,Z)
            nex =  numel(Z)/prod(nImgOut(this));
            ATY = zeros([nImgIn(this) nex],'like',Z); %start with transpose
            theta    = reshape(theta, [prod(this.sK(1:2)),this.sK(3:4)]);
            
            Yh = fft2(reshape(Z,[this.nImgOut nex]));
            for k=1:this.sK(3)
                Sk = reshape(this.S*squeeze(theta(:,k,:)),nImgOut(this));
                T  = Sk.*Yh;
                ATY(:,:,k,:) = squeeze(sum(T,3));
            end
            ATY = real(ifft2(ATY));
            ATY = reshape(ATY,[],nex);
        end
        
        function dY = Jthetamv(this,dtheta,~,Y,~)
            nex    =  numel(Y)/nFeatIn(this);
            Y      = reshape(Y,[],nex);
            dY = getOp(this,dtheta)*Y;
        end
        
        function dtheta = JthetaTmv(this,Z,~,Y)
            %  derivative of Z*(A(theta)*Y) w.r.t. theta
            
            nex    =  numel(Y)/nFeatIn(this);
            
            dth1    = zeros([this.sK(1)*this.sK(2),this.sK(3:4)],'like',Y);
            Y     = permute(reshape(Y,[nImgIn(this) nex ]),[1 2 4 3]);
            Yh    = reshape(fft2(Y),prod(this.nImg(1:2)),nex*this.sK(3));
            Zh    = permute(ifft2(reshape(Z,[nImgOut(this) nex])),[1 2 4 3]);
            Zh     = reshape(Zh,[], this.sK(4));
            
            for k=1:prod(this.sK(1:2)) % loop over kernel components
                temp = bsxfun(@times,conj(this.S(:,k)),Yh);
                temp = reshape(temp,[],this.sK(3));
                dth1(k,:,:) = conj(temp')*Zh;
            end
            dtheta = real(reshape(dth1,this.sK));
        end
    
  
        
    end
end


