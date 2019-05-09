function [ realization] = nonSimpat_Geology_ns( out,par)

    w_ssm = par.w_ssm;
    MR = par.MR;

    out_c = out;
    out = imresize(out_c,1/MR);
    % %     level = graythresh(out);
    % %     out = im2bw(out, level);
    par.Dimx = size(out,1);  par.Dimy = size(out,2);
    
    %% Initialize the empty grid
    par.szRealizationx  = par.Dimx  + (par.Pat  - 1);
    par.szRealizationy  = par.Dimy  + (par.Pat  - 1);
    par.szRealizationz = par.DimzAll;
    % set the initial value = -1 for the continious case; 0.5 for the binary case.
    realization       = -1*ones(par.szRealizationx, par.szRealizationy, par.szRealizationz);

    h = waitbar(0);  wb = 0;
    
    %% multi-resolution simulation 
    
    for m1 = MR:-1:1
        
        out = imresize(out_c,1/m1);
        par.Dimx = size(out,1);
        par.Dimy = size(out,2);
        % %         level = graythresh(out);
        % %         out = im2bw(out, level);
        if m1 ~= MR
            realization = imresize(realization,(m1+1)/m1);
            realization = +realization;
            %             level = graythresh(realization);
            %             realization = im2bw(realization, level);
            par.szRealizationx  = par.Dimx  + (par.Pat  - 1);
            par.szRealizationy  = par.Dimy  + (par.Pat  - 1);
            par.szRealizationz = par.DimzAll;
            difdx = (size(realization,1)  - par.szRealizationx)/2;
            difdy = (size(realization,2)  - par.szRealizationy)/2;
            realization = realization(floor(difdx)+1:end-ceil(difdx),floor(difdy)+1:end-ceil(difdy),:);
        end
    
        % Store the frozen nodes in each coarse simulation
        frozenRealiz = zeros(par.szRealizationx, par.szRealizationy, par.szRealizationz);
    
        [X, Locdb] = extractPatterns_non_ns(out,par);
        Locdb = Locdb + (par.Pat -1)/2;
    
        % Define a random path throught the grid nodes
        par.szCoarseGridx = fix((par.Dimx  - 1))+1;
        par.szCoarseGridy = fix((par.Dimy  - 1))+1;
        par.szCoarseGridz= fix((par.Dimz - 1))+1;
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
        wb = wb + 1;
        for i = 1:lengthRandomPath
            waitbar(i/lengthRandomPath*(1/par.MR)+(1/par.MR)*(wb-1),h,sprintf('stage%d : node:%5d : %3.0f%%',wb,i,100*i/lengthRandomPath));
        
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
                
                    Pattern    = X(idxNumber,:);
                
                case 'full'
                    if existNonFrozenNodes(frozenRealiz, wx(i,:), wy(i,:), wz(i,:))
                        dataLoc=[wx(i,1), wy(i,1)];
                        % calculate d_pat and d_loc and d_ns
                        idxNumber = findClosestPattern_Non(dataEvent(1,1:par.Pat^2*par.DimzFeat), X(:,1:par.Pat^2*par.DimzFeat), dataLoc, Locdb, weightEvent,w_ssm);
                    
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






end

