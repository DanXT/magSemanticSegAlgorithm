clc;clear all;close all;
if ispc
    load('..\studyarea\StudyArea.mat');
else
    % load('../../pre-process/5. modify incorrect interpretation/results/modifiedStudyArea.mat');
    load('../studyarea/StudyArea.mat');
end
% out1 = out1_n;
% out2 = out2_n;
subplot(2,2,1)
imagesc(out1)
subplot(2,2,2)
imagesc(out2)
subplot(2,2,3)
imagesc(geo_new)

% choose training area
uprow = 61;
rcol  = 151;
sizeTrA = 151;


ex1 = out1(uprow:end,1:rcol,:);
ex2 = out2(uprow:end,1:rcol,:);
ex_geo = geo_new(uprow:end,1:rcol);

figure
subplot(2,2,1)
imagesc(ex1)
subplot(2,2,2)
imagesc(ex2)
subplot(2,2,3)
imagesc(ex_geo)

out = cat(3, ex1, ex2, ex_geo);
save('result/exampler.mat','out');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choosing test area

testout1 = out1;
testout2 = out2;
testgeo_new = geo_new;



testout1 = testout1(:,rcol+1:end,:);
testout2 = testout2(:,rcol+1:end,:);
testgeo_new = testgeo_new(:,rcol+1:end,:);

figure;
subplot(2,2,1)
imagesc(testout1)
subplot(2,2,2)
imagesc(testout2)
subplot(2,2,3)
imagesc(testgeo_new)
outtest = cat(3, testout1, testout2, testgeo_new);

save('result/exampler_test.mat','outtest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
return