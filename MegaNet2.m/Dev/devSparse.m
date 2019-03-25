clear
close
nex = 1000;
nFeatIn =2000;
nFeatOut = 2000;
As = sprand(nFeatOut,nFeatIn,1e-1);
As = As~=0;
[I,J,~] = find(As);
Y = randn(nFeatIn,nex);
Z = randn(nFeatOut,nex);
t1 = nonzeros(((As').*(Y*Z'))');
t2 = nonzeros(As .* (Z*Y'));
norm(t1-t2)
tic;
t3 = 0;
for i=1:size(Y,2)
    t3 = t3 + Z(I,i) .* Y(J,i);
end
toc
norm(t1-t3)

%%
tic
t4 = convModMatVecMex(Z',Y',int32(I),int32(J));
toc
norm(t1-t4)
            