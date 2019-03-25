function Y = fft_3D(X)
            
Y = fft(fft(fft(X,[],1),[],2),[],3);
