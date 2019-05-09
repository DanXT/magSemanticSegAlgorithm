function [X, Locdb] = extractPatterns_non_ns(out,par)

% out : the exemplar
% Pat : template size
% m   : number of patterns to be skipped


DimzAll  = par.DimzAll;
Pat  = par.Pat;
Patz = par.Patz;
Dimx  = par.Dimx;   Dimy  = par.Dimy;
Dimz = par.Dimz;
m    = par.m;
FLIP = par.flip;

%% Training image 
%__________________________________________________________________________


% Number of patterns : disDim*disDim
disDimx  = ceil((Dimx  - (1+(Pat -1)) + 1)/m);
disDimy  = ceil((Dimy  - (1+(Pat -1)) + 1)/m);
disDimz = ceil((Dimz - (1+(Patz-1)) + 1)/m);
%fprintf('\n\nNext Phase\n------------------------------------------------\n');
fprintf('Number of Patterns : %d x %d x %d\n', disDimx,disDimy,disDimz);

%
Locdb=zeros(disDimx*disDimy*disDimz,2);
%
X=zeros(disDimx*disDimy*disDimz,Pat^2*DimzAll);

if FLIP
    Y=zeros(disDimx*disDimy*disDimz,Pat^2*DimzAll);   
    LocdbY=zeros(disDimx*disDimy*disDimz,2);
end



l=1;
for i=1:disDimx
    for j=1:disDimy
        for k=1:disDimz
            wx = 1+m*(i-1):1+m*(i-1)+(Pat -1);
            wy = 1+m*(j-1):1+m*(j-1)+(Pat -1);
            wz = 1:DimzAll;
            X(l,:)=reshape(out(wx,wy,wz),1,Pat^2*DimzAll);
                
            if FLIP
                ptch = out(wx,wy,wz);
                ptch2 = flip(ptch ,2);
                Y(l,:)=reshape(ptch2,1,Pat^2*DimzAll);
                LocdbY(l,:)=[wx(1),wy(1)];
            end
            %
            Locdb(l,:)=[wx(1),wy(1)];
            %
            l=l+1;
           
        end
    end
end

if FLIP
    X = [X;Y];
    Locdb = [Locdb;LocdbY];
end

end
