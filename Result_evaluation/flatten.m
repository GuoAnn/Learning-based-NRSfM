function [Xr] = flatten(X)
    x = X(1:3:end, :);
    y = X(2:3:end, :);
    z = X(3:3:end, :);
    Xr = [x(:)'; y(:)'; z(:)'];
end
