"""
离群点检测与过滤模块。
在 MATLAB 初始化之后、任何 Python 训练之前调用，一次性过滤所有数据结构。
"""
import numpy as np


def detect_outliers(normilized_point, Initial_shape, Gth=None, 
                    depth_deviation_ratio=0.3, gt_distance_threshold=None):
    """
    检测离群点。
    
    Parameters
    ----------
    normilized_point : (2F, N) 归一化 2D 坐标
    Initial_shape : (F, N) 初始深度值
    Gth : (F, 3, N) ground truth，可选
    depth_deviation_ratio : 深度偏离中位数的比例阈值
    gt_distance_threshold : GT 中距离中心过远的阈值，None 则自动计算
    
    Returns
    -------
    inlier_mask : (N,) bool array, True = 内点
    """
    num_frames = normilized_point.shape[0] // 2
    num_points = normilized_point.shape[1]
    
    # 确保 Initial_shape 是 (F, N)
    if Initial_shape.ndim == 1:
        depth_init = Initial_shape.reshape(num_frames, num_points)
    else:
        depth_init = Initial_shape
    
    outlier_reasons = {}
    
    # --- 条件 1：初始深度值为 0 或负值 ---
    # 参考 main_evaluation.m: idx = find(P1o(:,1)~=0)
    zero_depth_mask = np.any(np.abs(depth_init) < 1e-8, axis=0)  # (N,)
    outlier_reasons['zero_depth'] = int(np.sum(zero_depth_mask))
    
    # --- 条件 2：2D 坐标在所有帧中都为 (0,0) 的点 ---
    coord_zero_per_frame = np.zeros((num_frames, num_points), dtype=bool)
    for f in range(num_frames):
        u = normilized_point[2*f, :]
        v = normilized_point[2*f+1, :]
        coord_zero_per_frame[f] = (np.abs(u) < 1e-10) & (np.abs(v) < 1e-10)
    # 如果任何帧中 2D 坐标为零，标记为离群
    coord_zero_mask = np.any(coord_zero_per_frame, axis=0)
    outlier_reasons['zero_coord'] = int(np.sum(coord_zero_mask & ~zero_depth_mask))
    
    # --- 条件 3：深度值严重偏离中位数 ---
    # 参考 main_evaluation.m 中 norm(P1_old(j,:)-P(j,:))<100 的思路
    # 对每帧计算中位数，偏离过大则标记
    valid_depth = depth_init.copy()
    valid_depth[np.abs(valid_depth) < 1e-8] = np.nan
    median_depth = np.nanmedian(valid_depth, axis=1, keepdims=True)  # (F, 1)
    
    # 避免 median 为 0
    safe_median = np.where(np.abs(median_depth) < 1e-8, 1.0, median_depth)
    relative_deviation = np.abs(valid_depth - median_depth) / np.abs(safe_median)
    # 如果任何帧中偏离过大，标记
    deviation_mask = np.any(relative_deviation > depth_deviation_ratio, axis=0)
    # NaN 位置也标记 (来自 zero_depth)
    deviation_mask = deviation_mask | np.any(np.isnan(relative_deviation), axis=0)
    outlier_reasons['depth_deviation'] = int(np.sum(deviation_mask & ~zero_depth_mask & ~coord_zero_mask))
    
    # --- 条件 4：如果有 GT，GT 中 3D 坐标为 (0,0,0) 的点 ---
    gt_zero_mask = np.zeros(num_points, dtype=bool)
    if Gth is not None:
        if Gth.ndim == 2 and Gth.shape[0] == num_frames * 3:
            Gth_3d = Gth.reshape(num_frames, 3, num_points)
        elif Gth.ndim == 3:
            Gth_3d = Gth
        else:
            Gth_3d = None
        
        if Gth_3d is not None:
            # 任何帧中 GT 的 3D 坐标全为 0
            for f in range(num_frames):
                point_norm = np.sqrt(np.sum(Gth_3d[f] ** 2, axis=0))
                gt_zero_mask = gt_zero_mask | (point_norm < 1e-8)
            
            # GT 中距离中心过远的点
            if gt_distance_threshold is None:
                # 自动计算：基于所有帧 GT 的 IQR
                all_gt_norms = []
                for f in range(num_frames):
                    norms = np.sqrt(np.sum(Gth_3d[f] ** 2, axis=0))
                    all_gt_norms.append(norms)
                all_gt_norms = np.concatenate(all_gt_norms)
                valid_norms = all_gt_norms[all_gt_norms > 1e-8]
                if len(valid_norms) > 0:
                    q75, q25 = np.percentile(valid_norms, [75, 25])
                    iqr = q75 - q25
                    gt_distance_threshold = q75 + 3.0 * iqr  # 3 倍 IQR
            
            if gt_distance_threshold is not None:
                for f in range(num_frames):
                    centered = Gth_3d[f] - np.mean(Gth_3d[f], axis=1, keepdims=True)
                    point_dist = np.sqrt(np.sum(centered ** 2, axis=0))
                    gt_zero_mask = gt_zero_mask | (point_dist > gt_distance_threshold)
    
    outlier_reasons['gt_outlier'] = int(np.sum(gt_zero_mask & ~zero_depth_mask & ~coord_zero_mask & ~deviation_mask))
    
    # --- 合并所有条件 ---
    outlier_mask = zero_depth_mask | coord_zero_mask | deviation_mask | gt_zero_mask
    inlier_mask = ~outlier_mask
    
    print(f"\n{'='*60}")
    print(f"  Outlier Detection Report")
    print(f"{'='*60}")
    print(f"  Total points:        {num_points}")
    print(f"  Zero depth:          {outlier_reasons['zero_depth']}")
    print(f"  Zero 2D coord:       {outlier_reasons['zero_coord']}")
    print(f"  Depth deviation:     {outlier_reasons['depth_deviation']}")
    print(f"  GT outlier:          {outlier_reasons['gt_outlier']}")
    print(f"  Total outliers:      {int(np.sum(outlier_mask))} ({100*np.sum(outlier_mask)/num_points:.1f}%)")
    print(f"  Remaining inliers:   {int(np.sum(inlier_mask))}")
    print(f"{'='*60}\n")
    
    return inlier_mask


def filter_all_data(normilized_point, Initial_shape, Gth, J, inlier_mask):
    """
    根据 inlier_mask 过滤所有数据结构。
    
    Parameters
    ----------
    normilized_point : (2F, N)
    Initial_shape : (F, N)
    Gth : (F, 3, N)
    J : 包含 14 个 field 的 object，每个 field shape (F-1, N)
    inlier_mask : (N,) bool
    
    Returns
    -------
    过滤后的 normilized_point, Initial_shape, Gth, J
    """
    idx = np.where(inlier_mask)[0]
    N_new = len(idx)
    
    # 1. 过滤 normilized_point: (2F, N) → (2F, N_new)
    normilized_point_filtered = normilized_point[:, idx]
    
    # 2. 过滤 Initial_shape: (F, N) → (F, N_new)
    if Initial_shape.ndim == 1:
        num_frames = normilized_point.shape[0] // 2
        N = normilized_point.shape[1]
        depth_2d = Initial_shape.reshape(num_frames, N)
        Initial_shape_filtered = depth_2d[:, idx]
    elif Initial_shape.ndim == 2:
        Initial_shape_filtered = Initial_shape[:, idx]
    else:
        raise ValueError(f"Unexpected Initial_shape shape: {Initial_shape.shape}")
    
    # 3. 过滤 Gth: (F, 3, N) → (F, 3, N_new)
    Gth_filtered = None
    if Gth is not None:
        if Gth.ndim == 3:
            Gth_filtered = Gth[:, :, idx]
        elif Gth.ndim == 2:
            num_frames = normilized_point.shape[0] // 2
            N = normilized_point.shape[1]
            if Gth.shape[0] == num_frames * 3:
                Gth_3d = Gth.reshape(num_frames, 3, N)
                Gth_filtered = Gth_3d[:, :, idx]
            else:
                Gth_filtered = Gth[:, idx]
    
    # 4. 过滤 J 的所有 field: (F-1, N) → (F-1, N_new)
    J_fields = ['dx1_dy1', 'dx1_dy2', 'dx2_dy1', 'dx2_dy2',
                'ddx1_ddy1', 'ddx1_ddy2', 'ddx2_ddy1', 'ddx2_ddy2',
                'ddx1_dxdy', 'ddx2_dxdy',
                'dy1_dx1', 'dy2_dx1', 'dy1_dx2', 'dy2_dx2']
    
    for field_name in J_fields:
        if hasattr(J, field_name):
            data = getattr(J, field_name)
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == normilized_point.shape[1]:
                setattr(J, field_name, data[:, idx])
    
    print(f"Dataset filtered: {normilized_point.shape[1]} → {N_new} points")
    print(f"  normilized_point: {normilized_point.shape} → {normilized_point_filtered.shape}")
    print(f"  Initial_shape:    {Initial_shape.shape} → {Initial_shape_filtered.shape}")
    if Gth_filtered is not None:
        print(f"  Gth:              {Gth.shape} → {Gth_filtered.shape}")
    
    return normilized_point_filtered, Initial_shape_filtered, Gth_filtered, J