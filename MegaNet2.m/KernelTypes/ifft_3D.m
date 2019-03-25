function Y = ifft_3D(X)

Y = ifft(ifft(ifft(X,[],3),[],2),[],1);

