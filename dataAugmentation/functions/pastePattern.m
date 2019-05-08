function [realization, frozenRealiz] = pastePattern(Pattern, wx, wy, wz, realization, frozenRealiz, par, outT)

% This function will paste the pattern on the realization knowing the
% previous frozen nodes and and also updates the inner patch to become 
% frozen.
DimzAll = par.DimzAll;
Pat = par.Pat;
Patz= par.Patz;
innerPatch = par.innerPatch;
innerPatchz= par.innerPatchz;

frozenTemplate      = reshape(frozenRealiz(wx, wy, wz),1,Pat^2*DimzAll);
realizationTemplate = reshape(realization (wx, wy, wz),1,Pat^2*DimzAll);


% change the pattern values for frozen nodes
%---------------------------------------------------
Pattern(1,frozenTemplate == 1) = realizationTemplate(1,frozenTemplate == 1);



% Paste the pattern
%---------------------
% (one technique)
realization(wx, wy, wz) = reshape(Pattern, Pat, Pat, DimzAll);
% (second technique)
% for i = 1:Pat
%     for j = 1:Pat
%         for k = 1:Patz
%           realization(wx(i),wy(j),wz(k)) = Pattern(1, i+(j-1)*Pat+(k-1)*Pat^2);
%         end
%     end
% end



% Update the frozen nodes matrix
%------------------------------------
middleIdx  = (Pat +1)/2; 
middleIdxz = (Patz+1)/2;
boundary   = (innerPatch -1)/2;
boundaryz  = (innerPatchz-1)/2;
wxInner = wx(middleIdx- boundary  : middleIdx +boundary);
wyInner = wy(middleIdx- boundary  : middleIdx +boundary);
wzInner = wz;
frozenRealiz(wxInner, wyInner, wzInner) = 1;




end
