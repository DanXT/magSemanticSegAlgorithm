
clc; close all; clear;
addpath('functions');

% -------------------------------------------------------------------------
% Read Training Image
% -------------------------------------------------------------------------
if ispc
    load '..\prepareInput\result\exampler.mat';
else
    load '../prepareInput/result/exampler.mat';
end


% -------------------------------------------------------------------------
% Parameter Initialization
% -------------------------------------------------------------------------
opt.DimzAll = 7;
opt.DimzFeat= 6;
opt.Dimx  = size(out,1); % size of the exampler
opt.Dimy  = size(out,2);
opt.Dimz = 1;
opt.Pat  = 21;
opt.Patz = 1;
opt.innerPatch      = 9;
opt.innerPatchz     = 1;
opt.multipleGrid    = 3;
% m : number of patterns to be skipped
opt.m    = 4;
opt.flip = true;

numRe = 15;
DimzFeat =   [ 7     7  6  7     7   7   6     7  7    7    7    7  7  7    7];
Pat  =       [ 21   21 13 13    21  21  13    13 21    13   13  13 21  21  21 ];
innerPatch = [ 11   11  7  7    11  11   7     7 11    7    7   7  11  11  11];
w0 =        [ 0.1  0.3 0.5 0.5 0.5 0.7 0.8    0.8 0.8 0.5 0.5 0.5 0.5 0.5 0.5];
realization = zeros(opt.Dimx, opt.Dimy,opt.DimzAll,numRe);
for i=1:numRe
        opt.Pat = Pat(i);
        opt.DimzFeat = DimzFeat(i);
        opt.innerPatch = innerPatch(i);
        opt.w_ssm = w0(i);
       
        realization(:,:,:,i) = nonSimpat_Geology_ns(out, opt);
        
end


return
if ispc
    save('results\realizations.mat','realization','DimzFeat','Pat','w0','innerPatch')
else
    save('results/realizations.mat','realization','DimzFeat','Pat','w0','innerPatch')
end



