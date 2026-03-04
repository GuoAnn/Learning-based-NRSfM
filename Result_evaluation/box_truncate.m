% This function performs outlier detecting in a manner similar to boxplotting
% It truncates the upper quantile if it exceeeds a threshold based on the extend
% of the lower and upper middle quantile½
function [tr_d, outliers] = box_truncate(d)
    % Truncates outliers in the same style as Matlab's Boxplot function
    sort_field = sort(reshape(d, 1, []));
    % Find the 25% largest and %75 largest values in the field
    index_25 = round(size(sort_field, 2) * 0.25);
    index_75 = round(size(sort_field, 2) * 0.75);
    val_25 = sort_field(index_25);
    val_75 = sort_field(index_75);
    % Calculate the whisker limit
    w = 1.5 * (val_75 - val_25);
    tr_d = d;
    % Truncate!
    outliers = d > val_75 + w;
    tr_d(outliers) = val_75 + w;
end
