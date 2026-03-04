function [d, Xa, outliers] = nrsfm_score(X, Q)
    % Performs alignment of scoring of your reconstructed frame w.r.t
    % the supplied ground truth frame. Useful for assessing the accuracy of your
    % algorithm.
    %
    % X: Reconstruction of the NRSfM sequence
    % Q: Reference/groundtruth of the NRSfM sequence
    %
    % d: The resulting score/distance.
    % Xa: The supplied reconstruction after it has been aligned
    % outliers: The number of outliers that have been detected and truncated
    %

    % An inital guess for the metric optimization is needed
    % Pick the procrustes algorithm if there is the same number of points in X and Q
    guess = @procrustes_guess;
    % otherwise
    if ~all(size(X) == size(Q))
        % Zero-mean both shapes...
        mX = mean(flatten(X), 2);
        mQ = mean(flatten(Q), 2);
        X = X - repmat(mX, size(X, 1) / 3, size(X, 2));
        Q = Q - repmat(mQ, size(Q, 1) / 3, size(Q, 2));
        % and use ICP to find the initial guess
        guess = @icp_guess;
    end
    % Execute and find the initial rigid transform
    [s, R, t] = guess(X, Q);
    % Submit to the non-linear optimizer and find the optimal score
    p0 = [s reshape(R, 1, []) t];
    options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt');
    f = @(p) objective(p, X, Q);
    p = lsqnonlin(f, p0, [], [], options);
    % obtain the minimum metric score as well as the number of outliers...
    [d, outliers] = f(p);
    % And the final aligned shape
    Xa = transform(p, X);
    figure
    plot3(Xa(1,:),Xa(2,:),Xa(3,:),'.r');
    hold on;
    plot3(Q(1,:),Q(2,:),Q(3,:),'.g');
end
% Function for alignment guess when correspondences aren't known
% We assume that both shapes are zero-meaned
function [s, R, t] = icp_guess(X, Q)
    % Scale is obtained similarity to Porcrustes analysis
    s = scale(Q) / scale(X);
    % While translation and rotation is obtained via ICP
    [R, t] = icp(flatten(Q), s * flatten(X));
    t = t';
end
% We obtain alignment via procrustes if correspondences are known
function [s, R, t] = procrustes_guess(X, Q)
    Qr = flatten(Q);
    Xr = flatten(X);
    [~, ~, T] = procrustes(Qr', Xr');
    s = T.b;
    R = T.T';
    t = T.c(1, :);
end
% Estimates the scale of a given, zero-meaned shape
function [s] = scale(S)
    s = sqrt(sum(S(1:3:end).^2 + S(2:3:end).^2 + S(3:3:end).^2)) / size(S, 2);
end
% The objective function, which in brief transform the query shape
% and compares it to the reference
function [d, acc_outlier, com_outlier] = objective(p, X, Q)
    Xt = transform(p, X);
    [d, acc_outlier, com_outlier] = truncated_search(Xt, Q);
end
