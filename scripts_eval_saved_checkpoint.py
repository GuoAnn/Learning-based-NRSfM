import os, torch, numpy as np

RESULT_DIR = r"D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/results"  # 改成你的路径
UV_NP_PATH = os.path.join(RESULT_DIR, "Scene_normalized.npy")     # 你需要事先保存

shape_decoder_sd   = torch.load(os.path.join(RESULT_DIR, "Model_parameters.pt"), map_location="cpu")
shape_latent_code  = torch.load(os.path.join(RESULT_DIR, "shape_latent_code.pt"), map_location="cpu")
depth_saved        = torch.load(os.path.join(RESULT_DIR, "depth.pt"), map_location="cpu")
uv                 = np.load(UV_NP_PATH)          # [F,2,N]
F, _, N            = uv.shape
ones               = np.ones((F, 1, N), dtype=np.float32)
uv1                = np.concatenate([uv, ones], axis=1)
pts3d              = uv1 * depth_saved.cpu().numpy().repeat(3, 1)  # [F,3,N]

# 如果有 GT，可计算误差
GT_PATH = os.path.join(RESULT_DIR, "Pgth.npy")    # 可选：你也可以事先保存 Pgth
if os.path.exists(GT_PATH):
    gt = np.load(GT_PATH)                         # [F,3,N]
    rmse = [np.sqrt(np.sum((pts3d[f]-gt[f])**2, axis=0)).mean() for f in range(F)]
    print("Per-frame RMSE:", rmse, "Mean:", np.mean(rmse))

# 导出第0帧点云
import open3d as o3d
p = pts3d[0].T
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p)
o3d.io.write_point_cloud(os.path.join(RESULT_DIR, "recon_frame0.ply"), pcd)
print("Saved recon_frame0.ply")