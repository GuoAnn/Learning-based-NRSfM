import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import numpy as np
import torch
import matlab.engine

from Dataset.dataset_setting import dataset_params
from Dataset.load_dataset import normalized_points_without_downsample
from NRSfM_core.Initial_supervised_learning_multiple_model import Initial_supervised_learning
from NRSfM_core.train_shape_decoder import train_shape_decoder
from Result_evaluation.Shape_error import shape_error

# ============ 配置 ============

# 数据根路径（balloon）
DATA_ROOT = r"NRSfM_challenge\balloon\balloon"
# 读取的相机轨迹（可换成 flyby/line/semi_circle/tricky/zigzag）
CAMERA = "circle"
# 投影模型
PROJ = "perspective"

# 评分的 GT（已给出：gt_frame_25.txt）
GT_PATH = os.path.join(DATA_ROOT, "gt_frame_25.txt")  # 3 x P

# 结果保存路径
RESULT_DIR = r"NRSfM_challenge\results\balloon"
os.makedirs(RESULT_DIR, exist_ok=True)

# K^-1（与工程现有设定一致）
K_INV = dataset_params["K_inv"]  # [fx, fy]，默认 [528,528]

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 工具函数 ============

def load_W_MD(data_root, proj, camera):
    seq_txt = os.path.join(data_root, "sequence", proj, f"{camera}.txt")
    vis_txt = os.path.join(data_root, "visibility", proj, f"{camera}.txt")
    if not os.path.exists(seq_txt):
        raise FileNotFoundError(seq_txt)
    if not os.path.exists(vis_txt):
        raise FileNotFoundError(vis_txt)
    W = np.loadtxt(seq_txt)  # 2F x P
    MD = np.loadtxt(vis_txt) # 2F x P（布尔0/1）
    return W, MD, seq_txt, vis_txt

class JStruct(object):
    __slots__ = [
        'dx1_dy1','dx1_dy2','dx2_dy1','dx2_dy2',
        'ddx1_ddy1','ddx1_ddy2','ddx2_ddy1','ddx2_ddy2',
        'ddx1_dxdy','ddx2_dxdy',
        'dy1_dx1','dy2_dx1','dy1_dx2','dy2_dx2'
    ]

def compute_J_and_initdepth_with_matlab(meng, W_norm_2FxP):
    """
    使用仓库里的 MATLAB 初始化流程，计算 J 以及初始深度 Initial_shape（[F,N]）
    注意：这里直接调用 initialization_for_NRSfM_local_all_new（仓库已有）。
    """
    # W_norm_2FxP 是归一化的 2F x P（按像素除以 K_inv）
    W_mat = matlab.double(W_norm_2FxP.tolist())
    # 调用 MATLAB 初始化函数（该函数在仓库中）
    # 期望返回：J（含 14 个场的 struct）、InitialShape（F x N）
    # 若函数签名不同，请把这里的 nargout/接收变量按你仓库里的函数签名调整
    J_mat, InitialShape_mat = meng.initialization_for_NRSfM_local_all_new(W_mat, nargout=2)

    # 封装为 Python 侧的 JStruct
    J = JStruct()
    for name in J.__slots__:
        setattr(J, name, np.array(J_mat[name]))

    Initial_shape = np.array(InitialShape_mat)  # F x N
    return J, Initial_shape

def to_batched_F2N(W_2FxP):
    F = W_2FxP.shape[0] // 2
    P = W_2FxP.shape[1]
    Scene_normalized = np.zeros((F, 2, P), dtype=np.float32)
    for f in range(F):
        Scene_normalized[f, 0, :] = W_2FxP[2*f,   :]
        Scene_normalized[f, 1, :] = W_2FxP[2*f+1, :]
    return Scene_normalized

def reconstruct_points3d_from_depth(Scene_normalized_F2N, depth_F1N):
    F, _, N = Scene_normalized_F2N.shape
    pts3d = np.zeros((F, 3, N), dtype=np.float32)
    uv1 = np.concatenate([Scene_normalized_F2N, np.ones((F,1,N), dtype=np.float32)], axis=1)  # [F,3,N]
    depth_np = depth_F1N.detach().cpu().numpy().repeat(3, 1)  # [F,3,N]
    pts3d = uv1 * depth_np
    return pts3d

def save_reconstruction_txt(pts3d_F3N, out_txt):
    # 保存为 3F x P 文本，行序为：第0帧X、Y、Z；第1帧X、Y、Z；...
    F, _, P = pts3d_F3N.shape
    S = np.zeros((3*F, P), dtype=np.float64)
    for f in range(F):
        S[3*f+0, :] = pts3d_F3N[f, 0, :]
        S[3*f+1, :] = pts3d_F3N[f, 1, :]
        S[3*f+2, :] = pts3d_F3N[f, 2, :]
    np.savetxt(out_txt, S, fmt="%.7f")

# ============ 主流程 ============

def main():
    # 1) 读取 balloon 的序列与可见性
    W, MD, seq_txt, vis_txt = load_W_MD(DATA_ROOT, PROJ, CAMERA)  # 2F x P

    # 2) 归一化（按仓库既有写法）
    Scene_2FxP = normalized_points_without_downsample(W, K_INV)  # 2F x P
    Scene_F2N  = to_batched_F2N(Scene_2FxP)                      # [F,2,N]

    # 3) 启动 MATLAB，并计算 J 与初始深度
    m = matlab.engine.start_matlab()
    # 如果 nrsfm_score_v0.2 的 .m 文件不在 MATLAB path，可按需 addpath
    # m.addpath(r'NRSfM_challenge\nrsfm_score_v0.2', nargout=0)
    J, Initial_shape = compute_J_and_initdepth_with_matlab(m, Scene_2FxP)  # J: 14个场；Initial_shape: [F,N]

    # 4) 预训练导数网络（返回两个 DGCNNControlPoints 与 model_derivation）
    # 准备 normilized_point_batched（工程里 Initial_supervised_learning 期望的是 2F x N 的堆叠）
    normilized_point_batched = Scene_2FxP.copy()
    # 关键参数（与工程一致）
    kNN_degree = 20
    num_iterations = 2000
    num_data = 8

    shape_partial_derivate, model_derivation = Initial_supervised_learning(
        Initial_shape,                # [F,N]
        normilized_point_batched,     # 2F x N
        m,
        DEVICE,
        kNN_degree,
        num_iterations,
        num_data
    )

    # 5) 训练主网络（会输出 *.pt）
    # Gth：若没有每帧 GT，这里传零阵仅用于内部误差打印，不影响训练/保存
    F = Scene_F2N.shape[0]
    N = Scene_F2N.shape[2]
    Gth = np.zeros((F, 3, N), dtype=np.float32)

    # 结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    # 保存归一化观测，便于复现/评估
    np.save(os.path.join(RESULT_DIR, "Scene_normalized.npy"), Scene_F2N)

    # 训练
    train_shape_decoder(
        RESULT_DIR,
        normilized_point=Scene_F2N,   # [F,2,N]
        args=None,
        J=J,                          # 必须有 14 个字段
        m=m,
        Initial_shape=Initial_shape,  # [F,N]
        Gth=Gth,                      # [F,3,N]
        model_shape=shape_partial_derivate,
        model_derivation=model_derivation,
        device=DEVICE
    )

    # 6) 从保存的 depth.pt 反算 3D，并导出 3F x P 文本
    depth = torch.load(os.path.join(RESULT_DIR, "depth.pt"), map_location="cpu")  # [F,1,N]
    pts3d = reconstruct_points3d_from_depth(Scene_F2N, depth)                     # [F,3,N]
    out_txt = os.path.join(RESULT_DIR, f"reconstruction_{CAMERA}.txt")
    save_reconstruction_txt(pts3d, out_txt)
    print(f"已保存重建结果到 {out_txt}")

    # 7) 使用 gt_frame_25 评分（球面对齐+截断：使用 nrsfm_score 或工程内评估）
    if os.path.exists(GT_PATH):
        gt = np.loadtxt(GT_PATH)  # 3 x P
        # 0-index 的第 25 帧 -> MATLAB/数组第 26 行帧，但我们内部 F 是 0-based堆叠，这里直接取 f=25
        f = 25
        Sf = pts3d[f, :, :]  # 3 x P
        # 调用 nrsfm_score_v0.2（若已在 MATLAB path），否则用工程内的 shape_error（相对口径）
        try:
            # nrsfm_score 期望 X、Q 都是 3 x P，矩阵不需转置
            d, Xa, outliers = m.nrsfm_score(matlab.double(Sf.tolist()), matlab.double(gt.tolist()), nargout=3)
            d_val = float(np.array(d))
            print(f"[nrsfm_score] {CAMERA} 帧 {f} 距离 d = {d_val:.6f}")
            with open(os.path.join(RESULT_DIR, f"score_{CAMERA}.txt"), "w") as fw:
                fw.write(f"d={d_val:.6f}\n")
        except:
            # 回退：工程内误差（基于 draw_image_sparse），数值口径不同，仅供参考
            Gtmp = np.zeros_like(pts3d)
            Gtmp[f, :, :] = gt
            d_mean = shape_error(pts3d, Gtmp, m)
            print(f"[view_shape相对口径] {CAMERA} 平均误差 = {d_mean:.6f}")
            with open(os.path.join(RESULT_DIR, f"score_{CAMERA}.txt"), "w") as fw:
                fw.write(f"mean_error={d_mean:.6f}\n")
    else:
        print("未找到 GT 文件，跳过评分。")

if __name__ == "__main__":
    main()