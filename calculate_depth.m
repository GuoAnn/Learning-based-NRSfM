function P_grid=calculate_depth(N_res,u,v,par,point_used)
nC=40;
% 增强bending正则化：对第一帧使用更大的lambda以抑制伪曲面
lambdas_default = par*ones(nC-3, nC-3);
lambdas_strong  = par*10*ones(nC-3, nC-3); % 第一帧使用10倍的bending惩罚

P_grid = zeros(size(u,1),size(u,2));
for i=1:size(u,1)
    idx = point_used{i};
    umin=min(u(i,idx))-0.1; umax=max(u(i,idx))+0.1;
    vmin=min(v(i,idx))-0.1; vmax=max(v(i,idx))+0.1;
    bbsd = bbs_create(umin, umax, nC, vmin, vmax, nC, 1);
    colocd = bbs_coloc(bbsd, u(i,idx), v(i,idx));
    
    if i == 1
        % 第一帧：使用更强的bending正则化，抑制平面上的伪曲率
        bendingd = bbs_bending(bbsd, lambdas_strong);
    else
        bendingd = bbs_bending(bbsd, lambdas_default);
    end
    
    [ctrlpts3Dn]=ShapeFromNormals(bbsd,colocd,bendingd,[u(i,idx);v(i,idx);ones(1,length(u(i,idx)))],N_res(3*(i-1)+1:3*(i-1)+3,idx));
    mu=bbs_eval(bbsd, ctrlpts3Dn,u(i,idx)',v(i,idx)',0,0);
    P_grid(i,idx) = mu;
end
end