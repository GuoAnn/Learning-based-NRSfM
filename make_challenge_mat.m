function make_challenge_mat(dataset_root, proj, camera, K_inv, out_mat)
% 生成包含 J 和 scene 的 .mat 文件，供 Learning-based-NRSfM 使用
% 输入:
%   dataset_root: 例如 'NRSfM_challenge\balloon\balloon'
%   proj        : 'perspective' 或 'orthogonal'
%   camera      : 'circle' / 'flyby' / 'line' / 'semi_circle' / 'tricky' / 'zigzag'
%   K_inv       : [fx, fy]，默认 [528, 528]
%   out_mat     : 输出 .mat 路径，例如 'NRSfM_challenge\balloon\balloon\mat_file\balloon.mat'
%
% 依赖: 本仓库自带 BBS/ 目录，脚本会 addpath
% 说明: 
%   - J 的 14 个字段将按 (F-1)×N 维组织：每行对应从第 1 帧 → 第 k 帧（k=2..F）的映射导数。
%   - scene.m(i).m = 2×N，第 i 帧的 [u; v]（已按 K_inv 归一化）
%   - scene.Pgth(i).P = 3×N，全 0 占位（如有 per-frame GT 可自行填入）
%
% 作者: 你的工程内生成脚本，和仓库规范匹配

if nargin < 3, error('需要至少 dataset_root, proj, camera'); end
if nargin < 4 || isempty(K_inv), K_inv = [528, 528]; end
if nargin < 5 || isempty(out_mat)
    out_dir = fullfile(dataset_root, 'mat_file');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    out_mat = fullfile(out_dir, [camera '.mat']);
end

% 1) 读取 sequence 和 visibility
seq_txt = fullfile(dataset_root, 'sequence', proj, [camera '.txt']);
vis_txt = fullfile(dataset_root, 'visibility', proj, [camera '.txt']);
if ~exist(seq_txt, 'file'), error('找不到序列文件: %s', seq_txt); end
if ~exist(vis_txt, 'file'), error('找不到可见性文件: %s', vis_txt); end
W  = dlmread(seq_txt);   % 2F x N
MD = dlmread(vis_txt);   % 2F x N

[rows, N] = size(W);
if mod(rows, 2) ~= 0, error('W 行数应为 2F'); end
F = rows / 2;

% 2) 归一化
Scene_2FxN = zeros(size(W));
for f = 1:F
    Scene_2FxN(2*f-1, :) = W(2*f-1, :) / K_inv(1);
    Scene_2FxN(2*f,   :) = W(2*f,   :) / K_inv(2);
end

% 3) 组 scene 结构
scene = struct();
scene.m   = struct('m', cell(1, F));    % scene.m(i).m -> 2 x N
scene.Pgth= struct('P', cell(1, F));    % scene.Pgth(i).P -> 3 x N (占位)
for f = 1:F
    uv = [Scene_2FxN(2*f-1, :); Scene_2FxN(2*f, :)];
    scene.m(f).m = uv;
    scene.Pgth(f).P = zeros(3, N);      % 如有每帧 GT，可写入真实的 3×N
end

% 4) 组 J 结构（14 个字段）
J = struct();
fields = {'dx1_dy1','dx1_dy2','dx2_dy1','dx2_dy2', ...
          'ddx1_ddy1','ddx1_ddy2','ddx2_ddy1','ddx2_ddy2', ...
          'ddx1_dxdy','ddx2_dxdy', ...
          'dy1_dx1','dy2_dx1','dy1_dx2','dy2_dx2'};
for k = 1:numel(fields)
    J.(fields{k}) = zeros(max(F-1,1), N);
end

% 5) 计算从第 1 帧到每个第 k 帧 (k=2..F) 的 2D→2D 形变导数
addpath('./BBS');  % 使用仓库自带 BBS 工具
er = 1e-5; t = 1e-3; nC = 50;  % 和 fit_python.m 保持一致

% 第 1 帧的 2D 点 & 可见性
y1 = Scene_2FxN(1, :);   % u of frame 1
y2 = Scene_2FxN(2, :);   % v of frame 1
v1 = (MD(1, :) ~= 0) & (MD(2, :) ~= 0);

for k = 2:F
    % 第 k 帧的 2D 点 & 可见性
    y1_bar = Scene_2FxN(2*k-1, :);      % ubar of frame k
    y2_bar = Scene_2FxN(2*k,   :);      % vbar of frame k
    vk = (MD(2*k-1, :) ~= 0) & (MD(2*k, :) ~= 0);
    % 需要两帧都可见
    idx = find(v1 & vk);
    if numel(idx) < 6
        % 样本太少，跳过（留 0）
        continue;
    end

    % 以第 k 帧坐标 (ubar, vbar) 为自变量，拟合 x = f(y)，y=(ubar,vbar)，x=(u,v)
    umin = min(y1_bar(idx)) - t; umax = max(y1_bar(idx)) + t;
    vmin = min(y2_bar(idx)) - t; vmax = max(y2_bar(idx)) + t;
    bbs  = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);
    coloc= bbs_coloc(bbs, y1_bar(idx), y2_bar(idx));
    lambdas = er * ones(nC-3, nC-3);
    bending = bbs_bending(bbs, lambdas);

    % 拟合 x1 = f1(ybar)，最小二乘
    ctrl_u = (coloc' * coloc + bending) \ (coloc' * y1(idx)');  % u as target
    % 拟合 x2 = f2(ybar)
    ctrl_v = (coloc' * coloc + bending) \ (coloc' * y2(idx)');  % v as target

    % 在第 k 帧所有点的位置上评估导数（对不可见的点，结果后面置 0 即可）
    % 一阶
    dx1_dy1 = bbs_eval(bbs, ctrl_u', y1_bar', y2_bar', 1, 0)';   % ∂x1/∂y1
    dx1_dy2 = bbs_eval(bbs, ctrl_u', y1_bar', y2_bar', 0, 1)';   % ∂x1/∂y2
    dx2_dy1 = bbs_eval(bbs, ctrl_v', y1_bar', y2_bar', 1, 0)';   % ∂x2/∂y1
    dx2_dy2 = bbs_eval(bbs, ctrl_v', y1_bar', y2_bar', 0, 1)';   % ∂x2/∂y2
    % 二阶
    ddx1_ddy1 = bbs_eval(bbs, ctrl_u', y1_bar', y2_bar', 2, 0)'; % ∂²x1/∂y1²
    ddx1_ddy2 = bbs_eval(bbs, ctrl_u', y1_bar', y2_bar', 0, 2)'; % ∂²x1/∂y2²
    ddx1_dxdy = bbs_eval(bbs, ctrl_u', y1_bar', y2_bar', 1, 1)'; % ∂²x1/∂y1∂y2
    ddx2_ddy1 = bbs_eval(bbs, ctrl_v', y1_bar', y2_bar', 2, 0)'; % ∂²x2/∂y1²
    ddx2_ddy2 = bbs_eval(bbs, ctrl_v', y1_bar', y2_bar', 0, 2)'; % ∂²x2/∂y2²
    ddx2_dxdy = bbs_eval(bbs, ctrl_v', y1_bar', y2_bar', 1, 1)'; % ∂²x2/∂y1∂y2

    r = k-1;  % J 的行索引（从第 2 帧对应第 1 行开始）
    % 先清零
    J.dx1_dy1(r, :) = 0; J.dx1_dy2(r, :) = 0; J.dx2_dy1(r, :) = 0; J.dx2_dy2(r, :) = 0;
    J.ddx1_ddy1(r,:)= 0; J.ddx1_ddy2(r,:)= 0; J.ddx2_ddy1(r,:)= 0; J.ddx2_ddy2(r,:)= 0;
    J.ddx1_dxdy(r,:)= 0; J.ddx2_dxdy(r,:)= 0;
    J.dy1_dx1(r, :) = 0; J.dy1_dx2(r, :) = 0; J.dy2_dx1(r, :) = 0; J.dy2_dx2(r, :) = 0;

    % 填充可见点
    J.dx1_dy1(r, :) = dx1_dy1;
    J.dx1_dy2(r, :) = dx1_dy2;
    J.dx2_dy1(r, :) = dx2_dy1;
    J.dx2_dy2(r, :) = dx2_dy2;
    J.ddx1_ddy1(r,:)= ddx1_ddy1;
    J.ddx1_ddy2(r,:)= ddx1_ddy2;
    J.ddx2_ddy1(r,:)= ddx2_ddy1;
    J.ddx2_ddy2(r,:)= ddx2_ddy2;
    J.ddx1_dxdy(r,:)= ddx1_dxdy;
    J.ddx2_dxdy(r,:)= ddx2_dxdy;

    % 由一阶导数矩阵求逆得到 dy/dx
    % [dx1_dy1 dx1_dy2; dx2_dy1 dx2_dy2] * [dy1_dx1 dy1_dx2; dy2_dx1 dy2_dx2] = I
    detJ = dx1_dy1 .* dx2_dy2 - dx1_dy2 .* dx2_dy1;
    inv_ok = abs(detJ) > 1e-12;
    dy1_dx1 = zeros(1, N); dy1_dx2 = zeros(1, N);
    dy2_dx1 = zeros(1, N); dy2_dx2 = zeros(1, N);
    dy1_dx1(inv_ok) =  dx2_dy2(inv_ok) ./ detJ(inv_ok);
    dy1_dx2(inv_ok) = -dx1_dy2(inv_ok) ./ detJ(inv_ok);
    dy2_dx1(inv_ok) = -dx2_dy1(inv_ok) ./ detJ(inv_ok);
    dy2_dx2(inv_ok) =  dx1_dy1(inv_ok) ./ detJ(inv_ok);

    J.dy1_dx1(r, :) = dy1_dx1;
    J.dy1_dx2(r, :) = dy1_dx2;
    J.dy2_dx1(r, :) = dy2_dx1;
    J.dy2_dx2(r, :) = dy2_dx2;
end

% 6) 保存
save(out_mat, 'J', 'scene');
fprintf('Saved %s\n', out_mat);
end