function make_challenge_mat(dataset_root, proj, camera, K_inv, out_mat)
% 生成包含 J 和 scene 的 .mat 文件，供 Learning-based-NRSfM 使用
% 输入:
%   dataset_root: 例如 'NRSfM_challenge\balloon\balloon'
%   proj        : 'perspective' 或 'orthogonal'
%   camera      : 'circle' / 'flyby' / 'line' / 'semi_circle' / 'tricky' / 'zigzag'
%   K_inv       : [fx, fy]，默认 [528, 528]
%   out_mat     : 输出 .mat 路径，例如 'NRSfM_challenge\balloon\balloon\mat_file\balloon.mat'

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

fprintf('读取序列文件: %s\n', seq_txt);
W  = dlmread(seq_txt);   % 2F x N
MD = dlmread(vis_txt);   % 2F x N

[rows, N] = size(W);
if mod(rows, 2) ~= 0, error('W 行数应为 2F'); end
F = rows / 2;

fprintf('数据信息: F=%d, N=%d\n', F, N);

% 2) 归一化
Scene_2FxN = zeros(size(W));
for f = 1:F
    Scene_2FxN(2*f-1, :) = W(2*f-1, :) / K_inv(1);
    Scene_2FxN(2*f,   :) = W(2*f,   :) / K_inv(2);
end

% 3) 组 scene 结构
scene = struct();
m_cell = cell(1, F);
Pgth_cell = cell(1, F);

for f = 1:F
    uv = [Scene_2FxN(2*f-1, :); Scene_2FxN(2*f, :)];
    m_cell{f} = uv;
    Pgth_cell{f} = zeros(3, N);
end

scene.m = m_cell;
scene.Pgth = Pgth_cell;
scene.K = [384, 0, 320; 0, 384, 240; 0, 0, 1];

% 4) 组 J 结构
J = struct();
fields = {'dx1_dy1', 'dx1_dy2', 'dx2_dy1', 'dx2_dy2', ...
          'ddx1_ddy1', 'ddx1_ddy2', 'ddx2_ddy1', 'ddx2_ddy2', ...
          'ddx1_dxdy', 'ddx2_dxdy', ...
          'dy1_dx1', 'dy2_dx1', 'dy1_dx2', 'dy2_dx2'};

for k = 1:numel(fields)
    J.(fields{k}) = zeros(max(F-1, 1), N);
end

% 5) 检查BBS工具箱并计算导数
if exist('bbs_create', 'file') ~= 2
    error('BBS 工具箱未找到。请确保 BBS/ 目录在 MATLAB 路径中。');
end

er = 1e-5; t = 5e-3; nC_base = 50;  % 基础控制点数量

% 第1帧的2D点 & 可见性
y1 = Scene_2FxN(1, :);
y2 = Scene_2FxN(2, :);
v1 = (MD(1, :) ~= 0) & (MD(2, :) ~= 0);

fprintf('开始计算导数...\n');

for k = 2:F
    fprintf('处理帧 %d/%d...\n', k, F);
    
    % 第k帧的2D点 & 可见性
    y1_bar = Scene_2FxN(2*k-1, :);
    y2_bar = Scene_2FxN(2*k,   :);
    vk = (MD(2*k-1, :) ~= 0) & (MD(2*k, :) ~= 0);
    
    % 需要两帧都可见
    idx = find(v1 & vk);
    
    if numel(idx) < 10  % 增加最小点数要求
        fprintf('  警告: 可见点不足10个(%d个)，跳过导数计算\n', numel(idx));
        continue;
    end
    
    % 调试信息
    fprintf('  可见点数量: %d\n', numel(idx));
    
    try
        % 以第k帧坐标(ubar, vbar)为自变量，拟合x = f(y)
        umin = min(y1_bar(idx)) - t;
        umax = max(y1_bar(idx)) + t;
        vmin = min(y2_bar(idx)) - t;
        vmax = max(y2_bar(idx)) + t;
        
        % 检查范围是否有效
        if umax - umin < 1e-6 || vmax - vmin < 1e-6
            fprintf('  警告: 数据范围太小，跳过\n');
            continue;
        end
        
        fprintf('  拟合范围: u=[%.4f, %.4f], v=[%.4f, %.4f]\n', umin, umax, vmin, vmax);
        
        % 动态调整控制点数量：根据可见点数量确定
        nVisible = numel(idx);
        if nVisible < 30
            nC = 6;  % 对于少量点，使用较少的控制点
        elseif nVisible < 100
            nC = 10; % 中等数量的点
        else
            nC = min(15, floor(sqrt(nVisible/2))); % 大量点，但不超过15
        end
        
        % 确保控制点数量至少为4
        nC = max(4, nC);
        fprintf('  使用控制点数量: nC=%d (可见点=%d)\n', nC, nVisible);
        
        % 创建BBS结构
        bbs = bbs_create(umin, umax, nC, vmin, vmax, nC, 1);
        
        % 提取可见点的数据
        u_visible = y1_bar(idx);
        v_visible = y2_bar(idx);
        x1_visible = y1(idx);
        x2_visible = y2(idx);
        
        % 确保数据是列向量
        u_visible = u_visible(:);
        v_visible = v_visible(:);
        x1_visible = x1_visible(:);
        x2_visible = x2_visible(:);
        
        % 创建放置矩阵
        fprintf('  创建放置矩阵...\n');
        coloc_matrix = bbs_coloc(bbs, u_visible, v_visible);
        if isempty(coloc_matrix)
            error('bbs_coloc返回空矩阵');
        end
        
        % 检查矩阵维度是否匹配
        if size(coloc_matrix, 2) ~= nC * nC
            error('放置矩阵维度不匹配: 期望 %d 列，实际 %d 列', nC*nC, size(coloc_matrix, 2));
        end
        
        % 创建弯曲能量矩阵
        lambdas = er * ones(nC-3, nC-3);
        fprintf('  创建弯曲矩阵...\n');
        bending = bbs_bending(bbs, lambdas);
        
        % 拟合x1 = f1(ybar)
        fprintf('  拟合x方向...\n');
        ctrl_u = (coloc_matrix' * coloc_matrix + bending) \ (coloc_matrix' * x1_visible);
        ctrl_u = ctrl_u(:)';  % 转换为行向量
        
        % 拟合x2 = f2(ybar)
        fprintf('  拟合y方向...\n');
        ctrl_v = (coloc_matrix' * coloc_matrix + bending) \ (coloc_matrix' * x2_visible);
        ctrl_v = ctrl_v(:)';  % 转换为行向量
        
        % 检查控制点大小
        expected_size = nC * nC;
        if length(ctrl_u) ~= expected_size || length(ctrl_v) ~= expected_size
            fprintf('  警告: 控制点大小不正确。期望: %d, 实际: %d, %d\n', ...
                expected_size, length(ctrl_u), length(ctrl_v));
            continue;
        end
        
        % 在所有点位置评估导数
        u_all = y1_bar(:);
        v_all = y2_bar(:);
        
        fprintf('  计算导数...\n');
        
        % 一阶导数
        dx1_dy1 = bbs_eval(bbs, ctrl_u, u_all, v_all, 1, 0);
        dx1_dy2 = bbs_eval(bbs, ctrl_u, u_all, v_all, 0, 1);
        dx2_dy1 = bbs_eval(bbs, ctrl_v, u_all, v_all, 1, 0);
        dx2_dy2 = bbs_eval(bbs, ctrl_v, u_all, v_all, 0, 1);
        
        % 二阶导数
        ddx1_ddy1 = bbs_eval(bbs, ctrl_u, u_all, v_all, 2, 0);
        ddx1_ddy2 = bbs_eval(bbs, ctrl_u, u_all, v_all, 0, 2);
        ddx1_dxdy = bbs_eval(bbs, ctrl_u, u_all, v_all, 1, 1);
        ddx2_ddy1 = bbs_eval(bbs, ctrl_v, u_all, v_all, 2, 0);
        ddx2_ddy2 = bbs_eval(bbs, ctrl_v, u_all, v_all, 0, 2);
        ddx2_dxdy = bbs_eval(bbs, ctrl_v, u_all, v_all, 1, 1);
        
        r = k-1;
        
        % 填充导数矩阵
        J.dx1_dy1(r, :) = dx1_dy1;
        J.dx1_dy2(r, :) = dx1_dy2;
        J.dx2_dy1(r, :) = dx2_dy1;
        J.dx2_dy2(r, :) = dx2_dy2;
        J.ddx1_ddy1(r, :) = ddx1_ddy1;
        J.ddx1_ddy2(r, :) = ddx1_ddy2;
        J.ddx2_ddy1(r, :) = ddx2_ddy1;
        J.ddx2_ddy2(r, :) = ddx2_ddy2;
        J.ddx1_dxdy(r, :) = ddx1_dxdy;
        J.ddx2_dxdy(r, :) = ddx2_dxdy;
        
        % 计算逆导数
        detJ = dx1_dy1 .* dx2_dy2 - dx1_dy2 .* dx2_dy1;
        inv_ok = abs(detJ) > 1e-12;
        
        dy1_dx1 = zeros(1, N);
        dy1_dx2 = zeros(1, N);
        dy2_dx1 = zeros(1, N);
        dy2_dx2 = zeros(1, N);
        
        dy1_dx1(inv_ok) =  dx2_dy2(inv_ok) ./ detJ(inv_ok);
        dy1_dx2(inv_ok) = -dx1_dy2(inv_ok) ./ detJ(inv_ok);
        dy2_dx1(inv_ok) = -dx2_dy1(inv_ok) ./ detJ(inv_ok);
        dy2_dx2(inv_ok) =  dx1_dy1(inv_ok) ./ detJ(inv_ok);
        
        J.dy1_dx1(r, :) = dy1_dx1;
        J.dy1_dx2(r, :) = dy1_dx2;
        J.dy2_dx1(r, :) = dy2_dx1;
        J.dy2_dx2(r, :) = dy2_dx2;
        
        % 将不可见点的导数置零
        invisible_idx = ~(v1 & vk);
        if any(invisible_idx)
            J.dx1_dy1(r, invisible_idx) = 0;
            J.dx1_dy2(r, invisible_idx) = 0;
            J.dx2_dy1(r, invisible_idx) = 0;
            J.dx2_dy2(r, invisible_idx) = 0;
            J.ddx1_ddy1(r, invisible_idx) = 0;
            J.ddx1_ddy2(r, invisible_idx) = 0;
            J.ddx2_ddy1(r, invisible_idx) = 0;
            J.ddx2_ddy2(r, invisible_idx) = 0;
            J.ddx1_dxdy(r, invisible_idx) = 0;
            J.ddx2_dxdy(r, invisible_idx) = 0;
            J.dy1_dx1(r, invisible_idx) = 0;
            J.dy1_dx2(r, invisible_idx) = 0;
            J.dy2_dx1(r, invisible_idx) = 0;
            J.dy2_dx2(r, invisible_idx) = 0;
        end
        
        fprintf('  帧 %d 完成\n', k);
        
    catch ME
        fprintf('  处理帧 %d 时出错: %s\n', k, ME.message);
        fprintf('  错误发生在: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
        
        % 如果BBS相关函数出错，尝试跳过这一帧
        if contains(ME.message, 'bbs_') || contains(ME.message, 'BBS')
            fprintf('  BBS相关错误，跳过该帧\n');
            continue;
        else
            rethrow(ME);
        end
    end
end

% 6) 保存
fprintf('保存到: %s\n', out_mat);
save(out_mat, 'J', 'scene');

% 显示摘要
fprintf('\n完成! 生成的数据摘要:\n');
fprintf('  总帧数: %d\n', F);
fprintf('  点数: %d\n', N);
fprintf('  J矩阵大小: %d x %d\n', size(J.dx1_dy1, 1), size(J.dx1_dy1, 2));
fprintf('  scene.m: %d个单元格\n', length(scene.m));
fprintf('  scene.Pgth: %d个单元格\n', length(scene.Pgth));
end