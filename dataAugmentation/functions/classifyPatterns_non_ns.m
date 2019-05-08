function [X, Locdb] = classifyPatterns_non_ns(out,par)

% out : indicates the training image
% Pat : dimensio of the pattern template(better be odd value)
% m   : number of patterns to be skipped
% m1  : number of multiple-grids for this analysis
% way : indicates the distance function used for similarity measurement
% MDS : indicates the number of dimensions of MDS space
% clus: indicates the number of clusters for classification


global Pat;


% profile on
DimzAll  = par.DimzAll;
Pat  = par.Pat;
Patz = par.Patz;
Dimx  = par.Dimx;   Dimy  = par.Dimy;
Dimz = par.Dimz;
m    = par.m;
m1   = par.m1;
FLIP = par.flip;







%% Training image 
%__________________________________________________________________________


% Number of patterns : disDim*disDim
disDimx  = ceil((Dimx  - (1+(Pat -1)*m1) + 1)/m);
disDimy  = ceil((Dimy  - (1+(Pat -1)*m1) + 1)/m);
disDimz = ceil((Dimz - (1+(Patz-1)*m1) + 1)/m);
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
            %X((i-1)*disDim+j,:)=reshape(out(i:i+Pat-1,j:j+Pat-1),1,Pat^2);
            wx = 1+m*(i-1):m1:1+m*(i-1)+(Pat -1)*m1;
            wy = 1+m*(j-1):m1:1+m*(j-1)+(Pat -1)*m1;
            wz = 1:DimzAll;
           % % The "if" below is to delete completely empty patterns from calculations 
           % % you should also change to X initialization to X=[];
           % if sum(sum(out(wx,wy)))~=0
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
           % end
        end
    end
end

if FLIP
    X = [X;Y];
    Locdb = [Locdb;LocdbY];
end

end
