
Y = randn(24,14);
S = randn(24*14,9);
t = randn(9,1);
Z = randn(24,14);

a  = @(Y,t) real(vec(fft2(  ifft2(reshape(Y,24,14)).*reshape(S*t,24,14) )));
at = @(Y,Z) real(S'* vec(conj(ifft2(reshape(Y,24,14))) .* ifft2(reshape(Z,24,14))));

Z(:)'*a(Y,t)
t(:)'*at(Y,Z)