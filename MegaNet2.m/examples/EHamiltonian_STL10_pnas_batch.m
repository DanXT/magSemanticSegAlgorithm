nex  = [5000 500];
nt   = [4 3 2];
nfin = [32 24 16];

[NEX,NT,NFIN]= ndgrid(nex,nt,nfin);
for k=numel(NEX):-1:1
    EHamiltonian_STL10_pnas(NEX(k),NFIN(k),NT(k));
end
% function EHamiltonian_STL10_pnas(nex,nf0,nt)
