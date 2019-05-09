function [wx, wy, wz] = getPatternShape(node, par)

% This function gets the wx and wy  and wz which reflects the array where the
% pattern should be read from or written to. In other words, this array of
% wx and wy points to the location where the can be
% created from.
DimzAll      = par.DimzAll;
Pat           = par.Pat;

node = node';

% to account for created borders around realization
borders = (Pat  - 1)/2;   

% to make the grid be at center location of the template
r  = (Pat  + 1)/2;    

i  = 0:Pat -1;


wx = bsxfun(@times, ones(size(node,1), Pat),  1 + (node(:,1) -r) + borders);
wx = bsxfun(@plus, wx, i);
 
wy = bsxfun(@times, ones(size(node,1), Pat),  1 + (node(:,2) -r) + borders);
wy = bsxfun(@plus, wy, i);

wz = 1:DimzAll;
wz = repmat(wz,size(node,1),1);


end