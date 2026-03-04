import numpy as np
from scipy.spatial import KDTree

def box_truncate(d):
    """
    完全复现 MATLAB box_truncate.m 的逻辑
    """
    # 展平并排序
    d_flat = np.sort(d.flatten())
    n = len(d_flat)
    if n == 0: return d, 0

    # 计算索引 (Matlab round 0.25/0.75)
    idx_25 = int(round(n * 0.25)) - 1
    idx_75 = int(round(n * 0.75)) - 1
    idx_25 = max(0, idx_25)
    idx_75 = min(n - 1, idx_75)

    val_25 = d_flat[idx_25]
    val_75 = d_flat[idx_75]

    # Whisker 计算
    w = 1.5 * (val_75 - val_25)
    threshold = val_75 + w

    # 截断
    tr_d = d.copy()
    outliers = d > threshold
    tr_d[outliers] = threshold
    
    return tr_d, np.sum(outliers)

def solve_similarity_transform(X, Y):
    """
    求解最佳相似变换 (s, R, t) 使得 || s * X @ R + t - Y || 最小
    X, Y: (N, 3) 对应的点集
    """
    muX = X.mean(0)
    muY = Y.mean(0)
    
    X0 = X - muX
    Y0 = Y - muY
    
    # 1. 旋转 (Rotation)
    M = np.dot(X0.T, Y0)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    
    # 允许反射 (Reflection)，与 Matlab 保持一致
    # 如果不想允许反射，可以判断 det(R) < 0
    
    # 2. 尺度 (Scale)
    X0R = np.dot(X0, R)
    numer = np.sum(X0R * Y0)
    denom = np.sum(X0**2)
    s = numer / denom if denom > 1e-12 else 1.0
    
    # 3. 平移 (Translation)
    # t = muY - s * muX @ R
    t = muY - s * np.dot(muX, R)
    
    return s, R, t

def truncated_search(X, Q):
    """
    复现 MATLAB truncated_search.m
    使用 KDTree 计算双向 Chamfer 距离
    """
    # 建立 KDTree 加速搜索
    # 注意：Q 已经在之前被转置为 (N, 3)，符合 KDTree 输入
    tree_Q = KDTree(Q)
    tree_X = KDTree(X)
    
    # query(x) 返回 (distances, indices)
    # acc: X 中的每个点到 Q 中最近点的距离
    dist_acc, _ = tree_Q.query(X)
    
    # com: Q 中的每个点到 X 中最近点的距离
    dist_com, _ = tree_X.query(Q)
    
    # 截断 (Box Truncate)
    tr_acc, _ = box_truncate(dist_acc)
    tr_com, _ = box_truncate(dist_com)
    
    # 计算最终 Metric
    # MATLAB: d = (tr_acc' * tr_acc + tr_com' * tr_com) / (F * (N + M))
    # 这里 F=1 (单帧)
    sum_sq_err = np.sum(tr_acc**2) + np.sum(tr_com**2)
    n_points = len(X) + len(Q)
    
    d = np.sqrt(sum_sq_err / n_points)
    return d

def compute_challenge_error(pred_shape, gt_shape):
    """
    主入口：计算 Challenge Error
    流程：ICP 对齐 -> Truncated Search 评分
    """
    # 1. 数据准备 (N, 3)
    X = pred_shape.T.copy() # 预测
    Q = gt_shape.T.copy()   # GT
    
    # 2. 初始猜测：点对点 Procrustes (粗对齐)
    # 假设点序大体一致，先算一个初始姿态，避免 ICP 陷入局部最优
    s, R, t = solve_similarity_transform(X, Q)
    X = s * np.dot(X, R) + t
    
    # 3. ICP (Iterative Closest Point) 迭代精细对齐
    # 这一步模拟 MATLAB lsqnonlin 寻找最小化几何距离的过程
    max_icp_iter = 20
    prev_err = float('inf')
    
    for i in range(max_icp_iter):
        # A. 寻找对应点 (Find Correspondences)
        # 对于 X 中的每个点，找到 Q 中最近的点 Q_matched
        tree_Q = KDTree(Q)
        dist, indices = tree_Q.query(X)
        Q_matched = Q[indices]
        
        # B. 求解变换 (Solve Rigid/Sim Transform)
        # 将 X 对齐到 Q_matched
        s_icp, R_icp, t_icp = solve_similarity_transform(X, Q_matched)
        
        # C. 应用变换
        X = s_icp * np.dot(X, R_icp) + t_icp
        
        # D. 检查收敛 (可选)
        curr_err = np.mean(dist)
        if abs(prev_err - curr_err) < 1e-6:
            break
        prev_err = curr_err

    # 4. 计算最终分数 (使用双向 Chamfer 距离 + 截断)
    final_error = truncated_search(X, Q)
    
    return final_error