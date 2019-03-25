function [theta,W] = loadTensorFlowWeights(tffile,nb,nt)

if not(exist(tffile,'file'))
    error('cannot find file %s!',tffile);
else
    load(tffile);
end

flipt = @(T) vec(T);

thPy = [flipt(wK0); ...
    bias0'];
for j=0:nb-1
    for k=0:nt-1
        eval(sprintf('thPy = [thPy; flipt(u%d_b%d_wK1); flipt(u%d_b%d_wK2); vec(u%d_b%d_bias1); vec(u%d_b%d_bias2)];',j,k,j,k,j,k,j,k));
    end
    eval(sprintf('thPy = [thPy; flipt(u%d_b%d_wKconn); flipt(u%d_b%d_bias);];',j,nt,j,nt));
end
% thPy = [thPy; flipt(u0_b6_wKconn); vec(u0_b6_bias)];
WPy = [vec(wFC'); vec(biasFC)];
theta = thPy;
W = WPy;