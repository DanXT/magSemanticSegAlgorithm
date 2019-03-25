clc;clear all;close all;

if ispc

    load('..\..\data augmentation\1. prepareInput\result\exampler_test.mat');
else
    load('../../data augmentation/1. prepareInput/result/exampler_test.mat');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

patches = sampleIMAGES_ns(outtest, 64, 300);
for i = 1:25
    subplot(5,5,i)
    imagesc(uint8(patches(:,:,1:3,i)))
    axis off
end
figure
for i = 1:25
    subplot(5,5,i)
    imagesc(uint8(patches(:,:,4:6,i)))
    axis off
end
figure
for i = 1:25
    subplot(5,5,i)
    imagesc(uint8(patches(:,:,7,i)))
    axis off
end

%%%%%%%%%%%%%%%%%%%%


testPatches = patches;

return
if ispc
    save('results\testPatches.mat', 'testPatches');
else
    save('results/testPatches.mat', 'testPatches');
end