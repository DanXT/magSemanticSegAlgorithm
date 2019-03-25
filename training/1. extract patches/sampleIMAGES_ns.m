function [patches,imgIdx,ranPixx,ranPixy] = sampleIMAGES_ns(IMAGES, patchsize, numpatches)
% sampleIMAGES

[x,y,z,n] = size(IMAGES);
patches = zeros(patchsize,patchsize,z, numpatches);


imgIdx= randi(n,numpatches,1);
% randomly sample an 8�8 image patch from the selected image
ranPix1=randi(x-patchsize+1,numpatches,1);
ranPixx=ranPix1;


ranPix2=randi(y-patchsize+1,numpatches,1);
ranPixy=ranPix2;

for i=1:numpatches
    ptch=IMAGES(ranPixx(i):ranPixx(i)+patchsize-1, ranPixy(i):ranPixy(i)+patchsize-1,:,imgIdx(i));
    % convert the image patch into a 64-dimensional vector
    patches(:,:,:,i)=ptch;
end
