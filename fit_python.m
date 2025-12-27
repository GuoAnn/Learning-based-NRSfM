function [quv,dqu,dqv,ddqu,ddqv,ddquv]=fit(Image_2d,Point_3d,Points_evaluation_2d)
    addpath('./BBS/');%Use tool in BBS toolpack (Bicubic B-Splines)
    er = 1e-5;
    t= 1e-3;%A threshold to control the bound
    nC = 50;%A parameter to control the size of the grid total grid number is nC*nC
    idx = find(Image_2d(1,:)~=0);%Check whether some features are missing
    umax = max(Image_2d(1,idx))+t;%Seek the upper bound of x-axis direction
    umin = min(Image_2d(1,idx))-t;%Seek the lower bound of x-axis direction
    vmax = max(Image_2d(2,idx))+t;%Seek the upper bound of y-axis direction
    vmin = min(Image_2d(2,idx))-t;%Seek the lower bound of y-axis direction
    bbs = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);%Create a bidimensional cubic spline
    coloc = bbs_coloc(bbs, Image_2d(1,idx), Image_2d(2,idx));%B-spline collocation matrix
    lambdas = er*ones(nC-3, nC-3);
    bending = bbs_bending(bbs, lambdas);%B-spline bending matrix
    cpts = (coloc'*coloc + bending) \ (coloc'*Point_3d(1:3,idx)');%LLS
    ctrlpts = cpts';
    quv = bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',0,0);%get the value or derivation value of the B-spline
    dqu = bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',1,0);%get the value or derivation value of the B-spline
    dqv = bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',0,1);%1 0 mean the 1-order derivation value along x-axis
    ddqu = [];%bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',2,0);%get the value or derivation value of the B-spline
    ddqv = [];%bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',0,2);%get the value or derivation value of the B-spline
    ddquv = [];%bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',1,1);%get the value or derivation value of the B-spline
end