
n = 32; s = 3; nf = 64; ne = 1280;

larsConvFFT = (4*n^2*s^2*nf^2) + (4*n^2*nf^2 + 2*nf*n^2*log(n^2))*ne
obviousConv =  n^2*s^2*nf^2 *ne;
obviousFFT  =  n^2*log(n^2)*nf^2 + (4*n^2*nf^2 + 2*nf*n^2*log(n^2))*ne

obviousConv/obviousFFT
larsConvFFT/obviousFFT