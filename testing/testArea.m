% clc; 
clear all; close all;


nImg = [64 64];
cin = 6;
nClass = 3;
if ispc
load('results\testPatches_final.mat');

load('..\training\5. learning patches\result\weightsOpt.mat');

else
    load('../training/4. prepare patches for test/result/testPatches1.mat');

    load('../training/5. learning patches/result/weightsOpt.mat');
end

patches_test = reshape(Ytest, nImg(1), nImg(1), cin, size(Ytest,2));
% 

% scan the test area to get a number of patches (from other file)
% get labels for each labels
% put together to form a segment
% compare it with the original one
% computer the accuracy

%% show results
[YNk]        = apply(net,thOpt,Ytest);
szY  = size(YNk); nex  = szY(2);
Y = reshape(YNk, prod(nImg),[],nex);
Y = permute(Y,[2 1 3]);
Y = cat(1, Y, ones(1,prod(nImg),nex));
            
szW  = [nClass, size(Y,1)];
W    = reshape(WOpt,szW);
Y = reshape(Y,size(Y,1),[]);
WY = W*Y;
% make sure that the largest number in every row is 0
m = max(WY,[],1);
WY = WY - m;
S    = exp(WY);
Cp   = getLabels(pLoss,S);
[ro,co]  = find(Cp == 1);
Cpshow = ro;

Cpshow = reshape(Cpshow,[nImg,nex]);

clims = [1 3];
f1 = figure;
f2 = figure;
f3 = figure;
f4 = figure;
for i = 1:16
    figure(f1)
    subplot(4,4,i)
    imagesc(Cpshow(:,:,i),clims); axis off
    figure(f2)
    subplot(4,4,i)
    imagesc(squeeze(Ct(:,:,1,i)),clims); axis off
    figure(f3)
    subplot(4,4,i)
    imagesc(patches_test(:,:,1:3,i)); axis off
    figure(f4)
    subplot(4,4,i)
    imagesc(patches_test(:,:,4:6,i)); axis off;
end
save('results/predict.mat','Cpshow','Ct','patches_test');
return