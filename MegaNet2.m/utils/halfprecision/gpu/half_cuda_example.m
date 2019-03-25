MAX_FLOAT16_NUM = 65504;
MIN_FLOAT16_NUM = -65504;
A = randn(4000,4000,'single','gpuArray');
exp(10*randn(1,10))
A = max(MIN_FLOAT16_NUM,min(sign(A)*exp(3*A),MAX_FLOAT16_NUM));

format long
% A
Ah = Float2Half(A);

Ahf = Half2Float(Ah);

% Ahf

norm(A(:) - Ahf(:))/norm(A(:))