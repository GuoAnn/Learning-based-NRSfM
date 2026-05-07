"""
将过滤后的数据导出为新的 .mat 文件，供 MATLAB 可视化使用。
用法：
    python export_filtered_mat.py

需要修改下面的路径配置。
"""
import numpy as np
import scipy.io as sio
import os


def export_filtered_mat(original_mat_path, result_folder, output_mat_path=None):
    """
    读取原始 .mat 和 inlier_mask.npy，生成过滤后的 .mat 文件。
    """
    # 1. 加载 inlier_mask
    mask_path = os.path.join(result_folder, "inlier_mask.npy")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"找不到 inlier_mask.npy: {mask_path}\n"
                                f"请先运行一次 main.py 生成过滤结果。")
    
    inlier_mask = np.load(mask_path)  # (N,) bool
    inlier_idx = np.where(inlier_mask)[0]
    N_original = len(inlier_mask)
    N_filtered = len(inlier_idx)
    print(f"Inlier mask loaded: {N_original} -> {N_filtered} points")
    
    # 2. 加载原始 .mat —— 严格参照 load_dataset.py 的解析方式
    mat = sio.loadmat(original_mat_path)
    scene_mat = mat['scene']
    
    m_mat = scene_mat['m'][0, 0]        # (1, num_frames) object array
    Pgth_mat = scene_mat['Pgth'][0, 0]  # (1, num_frames) object array
    num_frames = m_mat.shape[1]
    
    # 验证点数一致
    num_points_check = m_mat[0, 0][0].shape[1]
    assert num_points_check == N_original, \
        f"点数不匹配: mat文件中 {num_points_check} vs inlier_mask中 {N_original}"
    
    print(f"Original data: {num_frames} frames, {N_original} points")
    
    # 3. 提取并过滤：构建简单的 numpy 数组
    # Scene_filtered: (2F, N')    —— 2D 归一化坐标
    # Pgth_filtered:  (F, 3, N')  —— 3D Ground Truth
    # depth 文件已经是 (F, N')，不需要再处理
    
    Scene_filtered = np.zeros((num_frames * 2, N_filtered), dtype=np.float64)
    Pgth_filtered = np.zeros((num_frames, 3, N_filtered), dtype=np.float64)
    
    for f in range(num_frames):
        # 参照 load_dataset.py 第 81-85 行的访问方式
        m_f = m_mat[0, f][0]       # (2, N) 或 (3, N)
        pgth_f = Pgth_mat[0, f][0] # (3, N)
        
        Scene_filtered[2 * f, :]     = m_f[0, inlier_idx]
        Scene_filtered[2 * f + 1, :] = m_f[1, inlier_idx]
        Pgth_filtered[f, 0, :]       = pgth_f[0, inlier_idx]
        Pgth_filtered[f, 1, :]       = pgth_f[1, inlier_idx]
        Pgth_filtered[f, 2, :]       = pgth_f[2, inlier_idx]
    
    # 4. 保存
    if output_mat_path is None:
        base_dir = os.path.dirname(original_mat_path)
        output_mat_path = os.path.join(base_dir, "matlab_filtered.mat")
    
    out = {
        'Scene_filtered': Scene_filtered,       # (2F, N') 2D坐标
        'Pgth_filtered': Pgth_filtered,         # (F, 3, N') 3D GT
        'num_points_filtered': N_filtered,
        'num_points_original': N_original,
        'num_frames': num_frames,
        'inlier_idx': inlier_idx + 1,           # MATLAB 是 1-based indexing
    }
    
    sio.savemat(output_mat_path, out, do_compression=True)
    print(f"Filtered .mat saved to: {output_mat_path}")
    print(f"  num_frames: {num_frames}")
    print(f"  num_points: {N_original} -> {N_filtered}")
    
    return output_mat_path


if __name__ == '__main__':
    # ========== 修改这里的路径 ==========
    original_mat_path = "/home/gax/NRSfM_dataset/dense_dataset/Dense_T-shirt/matlab.mat"
    result_folder     = "/home/gax/NRSfM_dataset/dense_dataset/Dense_T-shirt/results"
    # ====================================
    
    export_filtered_mat(original_mat_path, result_folder)
