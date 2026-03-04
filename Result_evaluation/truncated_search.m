% Function which calucates the distance metric between two sets of point tracks
function [d, acc_outlier, com_outlier] = truncated_search(X, Q)
    % Find the nearest neighbor for all points in X and Q
    [~, acc] = knnsearch(Q', X');
    [~, com] = knnsearch(X', Q');
    % Truncate the two set of distnaces with box plot truncation function
    [tr_acc, acc_outlier] = box_truncate(acc);
    [tr_com, com_outlier] = box_truncate(com);

    [F, N] = size(X); F = F / 3;
    [~, M] = size(Q);
    % Sum the truncated distances and normalize with respect to the number of points and frames
    d = (tr_acc' * tr_acc + tr_com' * tr_com) / (F * (N + M));
    d = sqrt(d);
end
