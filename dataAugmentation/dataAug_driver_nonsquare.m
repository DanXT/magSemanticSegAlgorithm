
clc; close all; clear;
addpath('functions');

% -------------------------------------------------------------------------
% Read Training Image
% -------------------------------------------------------------------------
if ispc
    load '..\1. prepareInput\result\exampler.mat';
else
    load '../1. prepareInput/result/exampler.mat';
end


% -------------------------------------------------------------------------
% Parameter Initialization
% -------------------------------------------------------------------------
par.DimzAll = 7;
par.DimzFeat= 6;
par.Dimx  = size(out,1); % size of the exampler
par.Dimy  = size(out,2);
par.Dimz = 1;
par.Pat  = 21;
par.Patz = 1;
par.innerPatch      = 9;
par.innerPatchz     = 1;

par.multipleGrid    = 3;

par.bShowMultiGrids = false;

par.newMG    = true;
%%%%%%%%%%%%%%%%%%%%
par.hardData = false;
%%%%%%%%%%%%%%%%%%%%



% histogram transformations
par.bTransCat    = false;
par.bTransCon    = false;

% m : number of patterns to be skipped
par.m    = 4;
par.way  = 1;
par.MDS  = 12;
par.clus = 100;
par.bUseKernelForClustering  = true;

par.bSkipPreviousFrozenNodes = true;

par.bDataEventOptimization   = false;
par.bUseDualTemplate         = false;
par.bPasteDualOnFrozen       = false;


% Load/Save files
par.bLoadVariables = false;
par.bSaveVariables = false;

par.flip = true;

numRe = 15;
DimzFeat =   [ 7     7  6  7     7   7   6     7  7    7    7    7  7  7    7];
Pat  =       [ 21   21 13 13    21  21  13    13 21    13   13  13 21  21  21 ];
innerPatch = [ 11   11  7  7    11  11   7     7 11    7    7   7  11  11  11];
w0 =        [ 0.1  0.3 0.5 0.5 0.5 0.7 0.8    0.8 0.8 0.5 0.5 0.5 0.5 0.5 0.5];
realization = zeros(par.Dimx, par.Dimy,par.DimzAll,numRe);
for i=1:numRe
        par.Pat = Pat(i);
        par.DimzFeat = DimzFeat(i);
        par.innerPatch = innerPatch(i);
        par.w_ssm = w0(i);
        realization(:,:,:,i) = nonSimpat_Geology_ns(out, par);

end


return
if ispc
    save('results\realizations.mat','realization','DimzFeat','Pat','w0','innerPatch')
else
    save('results/realizations.mat','realization','DimzFeat','Pat','w0','innerPatch')
end



