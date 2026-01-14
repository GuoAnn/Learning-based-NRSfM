# This is a sample Python script.
import os
# 建议：保持 allocator 配置，但不要指望它解决结构性 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

import argparse
import numpy as np
import glob
import torch
import matlab.engine

from Dataset.dataset_setting import dataset_params
from Dataset.load_dataset import normalized_points_downsample_load
from NRSfM_core.train_shape_decoder import train_shape_decoder, train_shape_decoder_GCN
from NRSfM_core.Initial_supervised_learning_multiple_model import Initial_supervised_learning
from NRSfM_core.new_DGCN_model import DGCNNControlPoints


def load_mat_dataset():
    file_path = "D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/KINECT_TSHIRT/mat_file/matlab.mat"
    files = glob.glob('D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/KINECT_TSHIRT/mat_file/*.mat')
    full_result_folder = os.path.join("D:/NRSfM/NIPS2022_Yongbo/results", dataset_params["results_base_folder"])
    try:
        os.mkdir(full_result_folder)
    except OSError:
        print("Creation of the directory %s failed" % full_result_folder)
    else:
        print("Successfully created the directory %s " % full_result_folder)

    Scene_normalized, Scene_apoints, J = normalized_points_downsample_load(file_path)
    np.save(os.path.join(full_result_folder, "Scene_normalized.npy"), Scene_normalized)
    np.save(os.path.join(full_result_folder, "Pgth.npy"), Scene_apoints)
    return full_result_folder, Scene_normalized, Scene_apoints, J, files


def main():
    parser = argparse.ArgumentParser(description='My first deep learning code for NRSfM')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (unused in GCN train loop)')
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs (unused)')
    parser.add_argument('--all_dataset', type=bool, default=False, help='whether use all dataset for training')

    # 新增：防 OOM 参数（你可从命令行改）
    parser.add_argument('--knn_k', type=int, default=8, help='kNN degree for graph (IMPORTANT for VRAM). Try 8/6/4')
    parser.add_argument('--num_data', type=int, default=2, help='num_data for Initial_supervised_learning (VRAM heavy)')
    parser.add_argument('--init_iters', type=int, default=10, help='iterations for Initial_supervised_learning')
    parser.add_argument('--gcn_iters', type=int, default=10, help='iterations for train_shape_decoder_GCN (default 50 in code)')
    parser.add_argument('--batch_frames', type=int, default=1, help='frames per batch inside loss (see loss_function.py batch_frames)')
    parser.add_argument('--use_amp', action='store_true', help='enable AMP in train_shape_decoder_GCN (safe wrapper)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 关键：MATLAB engine 放到 try/finally，确保退出
    m = matlab.engine.start_matlab()
    try:
        full_result_folder, Scene_normalized, Scene_apoints, J, file_id = load_mat_dataset()

        # 初始化深度
        Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_id[0], nargout=1))

        # 训练/加载 “第一个网络”（shape_partial_derivate 两个 DGCNNControlPoints）
        if dataset_params["save_or_load"] == "save":
            shape_partial_derivate, random_depth_data = Initial_supervised_learning(
                Initial_shape,
                Scene_normalized,
                m,
                device,
                kNN_degree=args.knn_k,
                num_iterations=args.init_iters,
                num_data=args.num_data
            )
        else:
            random_depth_data = []
            num_point_per_frame = Scene_normalized.shape[1]
            num_control_points = num_point_per_frame
            shape_partial_derivate = []
            shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=12, mode=0).to(device))
            shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=12, mode=0).to(device))

            PATH = os.path.join(full_result_folder, "0/model.pth")
            PATH1 = os.path.join(full_result_folder, "1/model1.pth")
            shape_partial_derivate[0].load_state_dict(torch.load(PATH, map_location=device))
            shape_partial_derivate[1].load_state_dict(torch.load(PATH1, map_location=device))

        # 保存模型（如果是 save）
        try:
            os.mkdir(os.path.join(full_result_folder, "0"))
            os.mkdir(os.path.join(full_result_folder, "1"))
        except OSError:
            pass

        if dataset_params["save_or_load"] == "save":
            torch.save(shape_partial_derivate[0].state_dict(), os.path.join(full_result_folder, "0/model.pth"))
            torch.save(shape_partial_derivate[1].state_dict(), os.path.join(full_result_folder, "1/model1.pth"))

        # 清理一次 GPU（避免第一个网络留下峰值缓存）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 进入第二个网络（这里是你 OOM 的地方）
        if random_depth_data:
            train_shape_decoder(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints,
                               shape_partial_derivate, random_depth_data, device)
        else:
            # 重要：把 args 传进去，用于控制 knn_k / AMP 等（需要配合下面的 train_shape_decoder_GCN 修改）
            train_shape_decoder_GCN(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints,
                                    shape_partial_derivate, device)
    finally:
        try:
            m.quit()
        except Exception:
            pass


if __name__ == '__main__':
    main()