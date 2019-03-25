clc;clear all;close all;
if ispc
    addpath('function\')
    load('..\3. extract test patches\results\testPatches.mat');
else
    addpath('function/')
load('../3. extract test patches/results/testPatches.mat');
end

[dimx, dimy, dimz, n1] = size(testPatches);
    
cin = 6;
nClass = 3;
    
Y0 = reshape(testPatches(:,:,1:cin,:),[], n1);
C = zeros(nClass,dimx,dimy,n1);
ctmp = testPatches(:,:,cin+1,:);
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


Ytest = Y0;
Ctest = C;
Ct    = ctmp;


if ispc
    save('result\testPatches1.mat','Ytest','Ctest','Ct');
else
    save('result/testPatches1.mat','Ytest','Ctest','Ct');
end
