function [ realization] = nonSimpat_Geology_ns( out,par)

w_ssm = par.w_ssm;
loop = par.multipleGrid;
patternIdx  = struct;
spaceString = '          '; 


% figure; hold on;


% Change the resolution for initial multiple-resolution setting

out_c = out;
out = imresize(out_c,1/loop); 
% %     level = graythresh(out);
% %     out = im2bw(out, level);
par.Dimx = size(out,1);  par.Dimy = size(out,2);
par.multipleGrid = 1;



% --------------------------------------------------------------------
% construct the realization grid
% --------------------------------------------------------------------
% make a bigger matrix so that the real realization be located in the
% middle of the grid and we have space for the boundry grids
par.szRealizationx  = par.Dimx  + (par.Pat  - 1)*2^(par.multipleGrid-1);
par.szRealizationy  = par.Dimy  + (par.Pat  - 1)*2^(par.multipleGrid-1);
par.szRealizationz = par.DimzAll;
% set the initial value = -1 for the continious case; 0.5 for the binary
% case.
realization       = 0.5*ones(par.szRealizationx, par.szRealizationy, par.szRealizationz);


%% ------------------------------------------------------------------------
% For each Coarse Grid Do:
%--------------------------------------------------------------------------
for m1 = loop:-1:1
    % change resolutions in multiple-resolution option
    out = imresize(out_c,1/m1); 
    par.Dimx = size(out,1);
    par.Dimy = size(out,2);
% %         level = graythresh(out);
% %         out = im2bw(out, level);
        if m1 ~= loop
            realization = imresize(realization,(m1+1)/m1);
            realization = +realization;
%             level = graythresh(realization);
%             realization = im2bw(realization, level);
            par.szRealizationx  = par.Dimx  + (par.Pat  - 1)*2^(par.multipleGrid-1);
            par.szRealizationy  = par.Dimy  + (par.Pat  - 1)*2^(par.multipleGrid-1);
            par.szRealizationz = par.DimzAll;
            difdx = (size(realization,1)  - par.szRealizationx)/2;
            difdy = (size(realization,2)  - par.szRealizationy)/2;
            realization = realization(floor(difdx)+1:end-ceil(difdx),floor(difdy)+1:end-ceil(difdy),:);
        end
        m1 = 1;
    
    
    
    par.m1 = 2^(m1-1);

    
    % Store the frozen nodes in each coarse simulation
    frozenRealiz = zeros(par.szRealizationx, par.szRealizationy, par.szRealizationz);

    
    
    
        % Classify Patterns by dissimilarity Matrix and MDS and & Kernel
        [X, Locdb] = classifyPatterns_non_ns(out,par);
        Locdb = Locdb + (par.Pat -1)/2;
%         [clusterModel, Z] = initializeKernelModel(Y, K, sigma, par);
        
%         clusterIdx = cell(par.clus,1);
%         clusterIdx = fillClusterIndex(clusterIdx, idx);
%         radonX     = calculateRadonX(X, par.Pat); %not adapted to 3D case
        fprintf('%s%s%s         ',spaceString,spaceString,spaceString);
        
    
    
    
    
    
    
    
    
    
   
    
    
    % Define a random path throught the grid nodes
    par.szCoarseGridx = fix((par.Dimx  - 1)/par.m1)+1;
    par.szCoarseGridy = fix((par.Dimy  - 1)/par.m1)+1;
    par.szCoarseGridz= fix((par.Dimz - 1)/par.m1)+1;
    lengthRandomPath = par.szCoarseGridx*par.szCoarseGridy*par.szCoarseGridz;
    randomPath       = randperm(lengthRandomPath);
    
    % Change to subscripts
    [nodeI, nodeJ, nodeK] = ind2sub([par.szCoarseGridx,par.szCoarseGridy,par.szCoarseGridz], randomPath);
    node                  = vertcat(nodeI,nodeJ,nodeK);
    [wx, wy, wz]          = getPatternShape(node, par);
    
    % ---------------------------------------------------------------------
    % Perform simulation
    % for each node in the random path Do:
    % ---------------------------------------------------------------------
    for i = 1:lengthRandomPath
        
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bnode: %5d  Percentage Completed: %3.0f%%',i,100*i/lengthRandomPath);
        
        
        if frozenRealiz(wx(i, (par.Pat+1)/2), wy(i, (par.Pat+1)/2), wz(i, (par.Patz+1)/2)) == 1
            continue
        end
        [dataEvent, status] = getDataEvent(realization, wx(i,:), wy(i,:), wz(i,:));
        
        
        weightEvent     = ones(1,par.Pat^2*par.DimzFeat);
        
        
        
        % Check if there is any data conditioning event or not and find the
        % pattern to be pasted on the simulation grid
        rng('default')
        switch status
            case 'empty'
                randIdx    = ceil(size(X,1).*rand(1,1));
                
                Pattern    = X(randIdx,:);
            case 'some'
                dataLoc=[wx(i,1), wy(i,1)];
                % calculate d_pat and d_loc and d_ns
                idxNumber = findClosestPattern_Non(dataEvent(1,1:par.Pat^2*par.DimzFeat), X(:,1:par.Pat^2*par.DimzFeat), dataLoc, Locdb, weightEvent,w_ssm);
                
                %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                Pattern    = X(idxNumber,:);
                
            case 'full'
                if existNonFrozenNodes(frozenRealiz, wx(i,:), wy(i,:), wz(i,:))
                    dataLoc=[wx(i,1), wy(i,1)];
                    % calculate d_pat and d_loc and d_ns
                    idxNumber = findClosestPattern_Non(dataEvent(1,1:par.Pat^2*par.DimzFeat), X(:,1:par.Pat^2*par.DimzFeat), dataLoc, Locdb, weightEvent,w_ssm);
                    %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                    Pattern    = X(idxNumber,:);
                else
                    continue
                end
        end
        
        % Paste the pattern on simulation grid and updates frozen nodes
        [realization, frozenRealiz] = pastePattern(Pattern, wx(i,:), wy(i,:), wz(i,:), realization, frozenRealiz, par, []);
    end
    
    
   
    
   
    
    
end




% crop the realization to its true dimensions
limitsx  = (par.szRealizationx  -par.Dimx )/2;
limitsy  = (par.szRealizationy  -par.Dimy )/2;
realization = realization(limitsx+1:limitsx+par.Dimx , limitsy+1:limitsy+par.Dimy , :);








fprintf('\n\nFinish!\n');





end

