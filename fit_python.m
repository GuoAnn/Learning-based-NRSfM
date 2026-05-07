function [quv,dqu,dqv,ddqu,ddqv,ddquv]=fit(Image_2d,Point_3d,Points_evaluation_2d)
    % Add path provided by user
    addpath('./BBS/'); 
    
    % --- [Modified] Robust Parameters for Tearing/Tricky Datasets ---
    % Original: er = 1e-5; nC = 50;
    % Reason: High nC + low er causes singular matrices in tearing regions (gaps).
    er = 0.5;   % significantly increased regularization to bridge gaps
    nC = 20;    % decreased grid size to prevent overfitting noise/tears
    t = 1e-3;   % Boundary threshold
    % ----------------------------------------------------------------
    
    idx = find(Image_2d(1,:)~=0); % Check valid points
    
    % Safety check: if too few points, return zeros to avoid crash
    if length(idx) < 10
        disp('Warning: Too few points for spline fitting.');
        num_eval = size(Points_evaluation_2d, 2);
        quv = zeros(3, num_eval);
        dqu = zeros(3, num_eval);
        dqv = zeros(3, num_eval);
        ddqu = zeros(3, num_eval);
        ddqv = zeros(3, num_eval);
        ddquv = zeros(3, num_eval);
        return;
    end

    umax = max(Image_2d(1,idx))+t;
    umin = min(Image_2d(1,idx))-t;
    vmax = max(Image_2d(2,idx))+t;
    vmin = min(Image_2d(2,idx))-t;
    
    % Create B-spline
    bbs = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);
    coloc = bbs_coloc(bbs, Image_2d(1,idx), Image_2d(2,idx));
    
    % Regularization matrix
    lambdas = er*ones(nC-3, nC-3);
    bending = bbs_bending(bbs, lambdas);
    
    % Construct Linear System: Ax = b
    A = coloc'*coloc + bending;
    b = coloc'*Point_3d(1:3,idx)';
    
    % --- [Modified] Fix Singularity / Hanging ---
    % Add a tiny value to the diagonal (Ridge Regression / Tikhonov Regularization)
    % This guarantees the matrix is invertible even if data has gaps (tearing).
    ridge_lambda = 1e-3; 
    A = A + eye(size(A)) * ridge_lambda;
    % --------------------------------------------
    
    % Solve
    cpts = A \ b;
    ctrlpts = cpts';
    
    % Evaluate
    quv = bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',0,0);
    dqu = bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',1,0);
    dqv = bbs_eval(bbs, ctrlpts, Points_evaluation_2d(1,:)',Points_evaluation_2d(2,:)',0,1);
    
    % Optional higher order derivatives (keep empty if original was empty)
    ddqu = []; 
    ddqv = [];
    ddquv = [];
end