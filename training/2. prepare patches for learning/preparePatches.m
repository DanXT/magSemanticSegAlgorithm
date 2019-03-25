clc;clear all;close all;

if ispc
    addpath('function\')
    load('..\1. extract patches\results\trainingPatches.mat');
else
    addpath('function/')
load('../1. extract patches/results/trainingPatches.mat');
end

[dimx, dimy, dimz, n1] = size(patches);
    
cin = 6;
nClass = 3;
    
Y0 = reshape(patches(:,:,1:cin,:),[], n1);
C = zeros(nClass,dimx,dimy,n1);
ctmp = patches(:,:,cin+1,:);
for i = 1:dimx
        for j = 1:dimy
            for k = 1:n1
                tind = ctmp(i,j,1,k);
                C(tind,i,j,k) = 1;
            end
        end
end
C  = reshape(C,[], n1);

    
Y0 = Y0./max(abs(Y0(:)));
    
Y0 = normalizeData(Y0')';





if ispc
    save('result\trainingData.mat','Y0','C','ctmp');
else
    save('result/trainingData.mat','Y0','C','ctmp');
end

