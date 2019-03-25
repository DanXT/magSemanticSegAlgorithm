% extract patches from the  exampler and realizations
% generate patches for training
clc;clear all;close all;
% load the exampler
if ispc
load('..\..\data augmentation\1. prepareInput\result\exampler.mat');

% load realizations
load('..\..\data augmentation\2. generate nonsquare realizations\results\realizations.mat');

else
    load('../../data augmentation/1. prepareInput/result/exampler.mat');

    % load realizations
    load('../../data augmentation/2. generate nonsquare realizations/results/realizations.mat');
end

realization(:,:,:,end+1) = out;

[x,y,z,n] = size(realization);

[patches, id_r, x_r, y_r] = sampleIMAGES_ns(realization, 64, 3000);
for i = 1:16
    subplot(4,4,i)
    imagesc(uint8(patches(:,:,1:3,i)));
    %title( [  int2str(id_r(i)), ':' ,  int2str(x_r(i)), '-',   int2str(y_r(i))   ]);
    axis off
end
figure
for i = 1:16
    subplot(4,4,i)
    imagesc(uint8(patches(:,:,4:6,i)))
    %title( [  int2str(id_r(i)), ':' ,  int2str(x_r(i)), '-',   int2str(y_r(i))   ]);
    axis off
end
figure
for i = 1:16
    subplot(4,4,i)
    imagesc(uint8(patches(:,:,7,i)))
    %title( [  int2str(id_r(i)), ':' ,  int2str(x_r(i)), '-',   int2str(y_r(i))   ]);
    axis off
end


for i = 1:n
    figure;
    imagesc(uint8(realization(:,:,1:3,i)));
    title(int2str(i))
end

for i = 1:n
    figure;
    imagesc(uint8(realization(:,:,7,i)));
    title(int2str(i))
end

return
if ispc
    save('results\trainingPatches.mat', 'patches');

else
    save('results/trainingPatches.mat', 'patches');
    
end

