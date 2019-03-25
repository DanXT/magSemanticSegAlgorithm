function [ realization] = nonSimpat_Geology_ns( out,par)

w_ssm = par.w_ssm;
loop = par.multipleGrid;
patternIdx  = struct;
spaceString = '          '; 


% figure; hold on;


% Change the resolution for initial multiple-resolution setting
if par.newMG
    out_c = out;
    m1 = 1;
    out = imresize(out_c,1/loop); 
% %     level = graythresh(out);
% %     out = im2bw(out, level);
    par.Dimx = size(out,1);  par.Dimy = size(out,2);
    par.multipleGrid = 1;
end



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





% --------------------------------------------------------------------
% assigning hard data
% --------------------------------------------------------------------
hardData = NaN(size(realization));
if par.hardData
    hardData = readHardData('hardData_neighboring.dat', hardData, par);
    
end




%% ------------------------------------------------------------------------
% For each Coarse Grid Do:
%--------------------------------------------------------------------------
for m1 = loop:-1:1
    
    

    
    % change resolutions in multiple-resolution option
    if par.newMG
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
    end
    
    
    par.m1 = 2^(m1-1);

    
    % Store the frozen nodes in each coarse simulation
    frozenRealiz = zeros(par.szRealizationx, par.szRealizationy, par.szRealizationz);

    
    
    if par.bLoadVariables
        
        fprintf('\n\nLoading variables from savedVar%d.mat ....  ',par.m1);
        load([dirName '\dist_pat\savedVar' num2str(par.m1) '.mat'],'-mat');
        fprintf('Done!\n%s%s%s         ',spaceString,spaceString,spaceString);
        
    else
        % Classify Patterns by dissimilarity Matrix and MDS and & Kernel
        [X, Locdb] = classifyPatterns_non_ns(out,par);
        Locdb = Locdb + (par.Pat -1)/2;
%         [clusterModel, Z] = initializeKernelModel(Y, K, sigma, par);
        
%         clusterIdx = cell(par.clus,1);
%         clusterIdx = fillClusterIndex(clusterIdx, idx);
%         radonX     = calculateRadonX(X, par.Pat); %not adapted to 3D case
        fprintf('%s%s%s         ',spaceString,spaceString,spaceString);
        
    end
    
    
    if par.bSaveVariables
        save([dirName '\dist_pat\savedVar' num2str(par.m1) '.mat'], 'X', 'Y', 'K', 'idx', 'prototype','sigma', 'clusterModel', 'Z','clusterIdx', 'radonX');
    end
    
    
    
    
    
    
   
    
    
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
    
    % test
    %for kq =1:2601
    %    if(nodeI(kq)==10 && nodeJ(kq)==10)
    %        aaaaa=kq;
    %    end
    %end
    
    
    % test 'moveHardData'
    % imshow(hardData);
    
    % fix the locations of hardData
    hardDataMoved = moveHardData(hardData, par);
    %if par.m1 == 2, hardDataMoved(67,65) = 0;hardDataMoved(69,65) = 1; end
    realization (~isnan(hardDataMoved)) = hardDataMoved(~isnan(hardDataMoved));
    frozenRealiz(~isnan(hardDataMoved)) = 1;
    
    % ---------------------------------------------------------------------
    % Perform simulation
    % for each node in the random path Do:
    % ---------------------------------------------------------------------
    for i = 1:lengthRandomPath
        
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bnode: %5d  Percentage Completed: %3.0f%%',i,100*i/lengthRandomPath);
        
        if par.bSkipPreviousFrozenNodes
            if frozenRealiz(wx(i, (par.Pat+1)/2), wy(i, (par.Pat+1)/2), wz(i, (par.Patz+1)/2)) == 1
                continue
            end
        end
        [dataEvent, status] = getDataEvent(realization, wx(i,:), wy(i,:), wz(i,:));
        
        if par.hardData
            wieghtEvent     = findWeight(hardDataMoved, frozenRealiz, wx(i,:), wy(i,:), wz(i,:));
        else
            wieghtEvent     = ones(1,par.Pat^2*par.DimzFeat);
        end
        
        
        % Check if there is any data conditioning event or not and find the
        % pattern to be pasted on the simulation grid
        rand('twister', sum(100*clock));
        switch status
            case 'empty'
                randIdx    = ceil(size(X,1).*rand(1,1));
                if par.bUseDualTemplate
                    patternIdx = findRangeDualTemplate(randIdx, par);
                end
                Pattern    = X(randIdx,:);
            case 'some'
                dataLoc=[wx(i,1), wy(i,1)];
                % calculate d_pat and d_loc and d_ns
                idxNumber = findClosestPattern_Non(dataEvent(1,1:par.Pat^2*par.DimzFeat), X(:,1:par.Pat^2*par.DimzFeat), dataLoc, Locdb, wieghtEvent,w_ssm);
                
                %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                Pattern    = X(idxNumber,:);
                
                %%%hd_condition = hardDataMoved(wx(i,:), wy(i,:), wz(i,:));
                %%%Pattern(~isnan(hd_condition))=hd_condition(~isnan(hd_condition));
%                 idxNumber = findClosestCluster(dataEvent, prototype);
%                 [Pattern, cluster]   = createPattern(idxNumber, clusterModel, Z, X, Y, idx, Pat, cluster,radonX);
%                 Pattern   = reshape(Pattern, 1, Pat^2);
            case 'full'
                if existNonFrozenNodes(frozenRealiz, wx(i,:), wy(i,:), wz(i,:))
                    dataLoc=[wx(i,1), wy(i,1)];
                    % calculate d_pat and d_loc and d_ns
                    idxNumber = findClosestPattern_Non(dataEvent(1,1:par.Pat^2*par.DimzFeat), X(:,1:par.Pat^2*par.DimzFeat), dataLoc, Locdb, wieghtEvent,w_ssm);
                    %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                    Pattern    = X(idxNumber,:);
                    
                    %%%hd_condition = hardDataMoved(wx(i,:), wy(i,:), wz(i,:));
                    %%%Pattern(~isnan(hd_condition))=hd_condition(~isnan(hd_condition));
                    %idxNumber = findClosestCluster(dataEvent, prototype, wieghtEvent);
                    %[Pattern, patternIdx] = findClosestInCluster(dataEvent, X, clusterIdx{idxNumber}, par,radonX, wieghtEvent);
                else
                    continue
                end
        end
        
        % for simplicity. use values instead of variables.
        
        
        

        % Paste the pattern on simulation grid and updates frozen nodes
        if par.bUseDualTemplate
            [realization, frozenRealiz] = pastePattern(Pattern, wx(i,:), wy(i,:), wz(i,:), realization, frozenRealiz, par, out(patternIdx.x,patternIdx.y, patternIdx.z));
        else
            [realization, frozenRealiz] = pastePattern(Pattern, wx(i,:), wy(i,:), wz(i,:), realization, frozenRealiz, par, []);
        end
        
       
        
        
    end
    
    
    % show results at each multiple-grid
    if par.bShowMultiGrids
        if par.Dimz > 1 %3D case
            slice3d(realization((par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim,(par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim));
            pause(0.1);
        else
            colormap(copper);   %2D
            imagesc(realization((par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim,(par.szRealization-par.Dim)/2+1:(par.szRealization-par.Dim)/2+par.Dim));
            axis square off;
            drawnow expose;
            pause(0.1);
        end
    end
    
    % to do TRANSCAT in the penultimate multigrids
    if par.bTransCat || par.bTransCon
        if par.m1 == 2
            limits  = (par.szRealization  -par.Dim )/2;
            limitsz = (par.szRealizationz -par.Dimz)/2;
            if ~par.bUseDualTemplate
                real     = realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                harddata = hardData   (limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                if par.bTransCon
                    real = histeq(real(1:par.m1:end,1:par.m1:end,1:par.m1:end),hist(out(:),par.Dim^2*par.Dimz));
                else
                    real = transcat(real(1:par.m1:end,1:par.m1:end,1:par.m1:end), out, harddata(1:par.m1:end,1:par.m1:end,1:par.m1:end), par, 1);
                end
                realization(limits+1:par.m1:limits+par.Dim , limits+1:par.m1:limits+par.Dim , limitsz+1:par.m1:limitsz+par.Dimz) = real;
            else
                real     = realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                harddata = hardData   (limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);
                if par.bTransCon
                    real = histeq(real,hist(out(:),par.Dim^2*par.Dimz));
                else
                    real = transcat(real, out, harddata, par, 1);
                end
                realization(limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz) = real;
            end
        end
    end
    
    
end




% crop the realization to its true dimensions
limitsx  = (par.szRealizationx  -par.Dimx )/2;
limitsy  = (par.szRealizationy  -par.Dimy )/2;
limitsz = (par.szRealizationz -par.Dimz)/2;
realization = realization(limitsx+1:limitsx+par.Dimx , limitsy+1:limitsy+par.Dimy , :);
% hardData    = hardData   (limits+1:limits+par.Dim , limits+1:limits+par.Dim , limitsz+1:limitsz+par.Dimz);



% TRANSCAT (Transformation of categorical proportions of realization to TI proportions)
if par.bTransCat
    realization_C = transcat(realization, out, hardData, par, 1);
    % show the transformed version of the realization too
    figure; imagesc(realization_C);
    axis square; axis xy off; colormap copper;
end



fprintf('\n\nFinish!\n');





end

