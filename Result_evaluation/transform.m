% Performs rigid transform of a shape with the given vectorized s R and t parameters
function [Xt] = transform(p, X)
    s = p(1);
    R = reshape(p(2:10), 3, 3);
    t = p(11:13)';

    [frames, points] = size(X); frames = frames / 3;

    %Xt = s * R * X + t;
    Xt = s * kron(eye(frames), R) * X + kron(ones(frames, points), t);
end
