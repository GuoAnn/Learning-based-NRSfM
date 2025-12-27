function P_grid=calculate_depth(N_res,u,v,par,point_used)
nC=40;
lambdas = par*ones(nC-3, nC-3);
P_grid = zeros(size(u,1),size(u,2));
for i=1:size(u,1)
     idx = point_used{i};
%    idx = find(u(i,:)~=100) & find(v(i,:)~=100);
    umin=min(u(i,idx))-0.1;umax=max(u(i,idx))+0.1;
    vmin=min(v(i,idx))-0.1;vmax=max(v(i,idx))+0.1;
    bbsd = bbs_create(umin, umax, nC, vmin, vmax, nC, 1);
    colocd = bbs_coloc(bbsd, u(i,idx), v(i,idx));
    bendingd = bbs_bending(bbsd, lambdas);
    [ctrlpts3Dn]=ShapeFromNormals(bbsd,colocd,bendingd,[u(i,idx);v(i,idx);ones(1,length(u(i,idx)))],N_res(3*(i-1)+1:3*(i-1)+3,idx));
    mu=bbs_eval(bbsd, ctrlpts3Dn,u(i,idx)',v(i,idx)',0,0);
    P_grid(i,idx) =mu;
    
end
