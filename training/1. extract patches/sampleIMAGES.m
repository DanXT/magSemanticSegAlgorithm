function patches = sampleIMAGES(IMAGES, patchsize, numpatches)
% sampleIMAGES

[x,~,z,n] = size(IMAGES);
patches = zeros(patchsize,patchsize,z, numpatches);


imgIdx= randi(n,numpatches,1);
% randomly sample an 8�8 image patch from the selected image
ranPix=randi(x-patchsize+1,numpatches,2);
ranPixx=ranPix(:,1);
ranPixy=ranPix(:,2);

for i=1:numpatches
    ptch=IMAGES(ranPixx(i):ranPixx(i)+patchsize-1, ranPixy(i):ranPixy(i)+patchsize-1,:,imgIdx(i));
    % convert the image patch into a 64-dimensional vector
    patches(:,:,:,i)=ptch;
end










%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
%patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
