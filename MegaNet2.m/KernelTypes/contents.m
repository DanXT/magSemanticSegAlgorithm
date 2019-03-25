% MegaNet2/kernels
% 
% Kernels are transformations applied to the feature vectors, writen as
%
%   K(theta,Y)
% 
% Examples: 
% 
% (affine):               theta_1 * Y    + B*theta_2
% (antisym affine):       (theta_1-theta_1') * Y    + B*theta_2
% (convolution):          K(theta_1) * Y + B*theta_2