function PCop = getThetaPC(fctn,d2YF,theta,Yk,tmp)

spcg = steihaugPCG;
spcg.maxIter=5;

nb = numel(fctn.net.blocks);

regOps = fctn.pRegTheta.B.blocks;
blocks = fctn.net.blocks;

PC = cell(1,1);
cnt = 0;
alphak = fctn.pRegTheta.alpha;

for k=1:nb
    nthk = nTheta(blocks{k});
    thk = theta(cnt+(1:nthk));
    thr = theta(cnt+nthk+1:end);
	if nthk==0
		PC{k} = opEye(0);
	else
	    Jth = getJthetaOp(blocks{k},thk,tmp{k}{1,1},tmp{k});
	    
	    Regk = regOps{k};
	    if k<nb
	        netk = MegaNet(blocks(k+1:nb));
	        Jy  = getJYOp(netk,thr,tmp{k}{end,1},tmp(k+1:end));
	        Hk = Jth'*Jy'*d2YF*Jy*Jth + alphak*Regk'*Regk;
	    else
	        Hk = Jth'*d2YF*Jth + alphak*Regk'*Regk;
	    end
	    
	    regPC = getPCop(Regk);
	    fctn = @(x) solve(spcg,Hk,x,[],regPC);
	    PC{k} = LinearOperator(nthk,nthk,fctn,fctn);
	end    
    cnt = cnt + nthk;
end
PCop = opBlkdiag(PC{:});

end
