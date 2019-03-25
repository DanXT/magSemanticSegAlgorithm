nex  = [5000 500];
nt   = [8 16 24];
nfin = [16 24 32];
nfout = [32 32];

[NEX,NT,NFIN,NFOUT]= ndgrid(nex,nt,nfin,nfout);
% function EParabolic_STL10_pnas2(nex,nt,nfin,nfout)
for k=numel(NEX):-1:1
    EParabolic_STL10_pnas2(NEX(k),NT(k),NFIN(k),NFOUT(k));
end