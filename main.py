# This is a sample Python script.
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import argparse
import numpy as np
#import scipy.io
import matlab.engine
import glob
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Dataset.dataset_setting import dataset_params
from Dataset.result_setting import result_params
from Dataset.load_dataset import load_preprocessed_W,normalized_points_downsample,normalized_points_without_downsample,normalized_points_downsample_load
from NRSfM_core.train_shape_decoder import train_shape_decoder, train_shape_decoder_GCN
from NRSfM_core.Initial_supervised_learning_DGCN import Initial_supervised_learning_DGCN
#from NRSfM_core.Initial_supervised_learning_framework import Initial_supervised_learning
#from NRSfM_core.Initial_supervised_learning_two_models import Initial_supervised_learning
from NRSfM_core.Initial_supervised_learning_multiple_model import Initial_supervised_learning
from NRSfM_core.Collect_datasets import Collect_data, Initial_learning_from_all_datasets
from NRSfM_core.new_DGCN_model import DGCNNControlPoints

import torch

m = matlab.engine.start_matlab()

def load_mat_dataset():
    file_path="D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/KINECT_TSHIRT/mat_file/matlab.mat"
    files = glob.glob('D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/KINECT_TSHIRT/mat_file/*.mat')
    full_result_folder = os.path.join("D:/NRSfM/NIPS2022_Yongbo/results",dataset_params["results_base_folder"])
    # 拼接完整的结果文件夹路径，将三个部分组合成一个路径
    try:
        os.mkdir(full_result_folder)
    except OSError:
        print("Creation of the directory %s failed" % full_result_folder)
    else:
        print("Successfully created the directory %s " % full_result_folder)
    Scene_normalized, Scene_apoints, J = normalized_points_downsample_load(file_path)
    # Scene_normalized: [F,2,N] 归一化的 (u,v)
    # Scene_apoints:    [F,3,N] Ground truth 3D（如数据有）
    # J: image warp 的一阶/二阶导数（用于局部微分约束）
    np.save(os.path.join(full_result_folder, "Scene_normalized.npy"), Scene_normalized) #保存归一化的2D点
    np.save(os.path.join(full_result_folder, "Pgth.npy"),Scene_apoints) #保存3D点
    return full_result_folder, Scene_normalized, Scene_apoints, J, files #files根本没用

'''
def load_dataset():#这才是完整的加载数据集函数，但是mat部分设置感觉有点问题
    # Create a folder to save reconstructed results
    full_result_folder = os.path.join("D:/NRSfM/NIPS2022_Yongbo/results",dataset_params["results_base_folder"])
    try:
        os.mkdir(full_result_folder)
    except OSError:
        print("Creation of the directory %s failed" % full_result_folder)
    else:
        print("Successfully created the directory %s " % full_result_folder)
    # Load the 2D measurements
    full_W_folder = os.path.join("D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/",dataset_params["preprocessed_W_filename"])
    W_np = load_preprocessed_W(full_W_folder, np.float32)
    if dataset_params["downsample"]:
        if dataset_params["load_data"]:
            full_Mat_folder = dataset_params["full_Mat_folder"]
            W_normalized, A, J = normalized_points_downsample_load(full_Mat_folder)
        else:
            W_normalized = normalized_points_downsample(W_np, dataset_params["K_inv"],dataset_params["downsample_size"])
            J=0
    else:
        W_normalized=normalized_points_without_downsample(W_np,dataset_params["K_inv"])
        J=0

    return full_result_folder, W_normalized, W_np, J
    '''

def load_mat_all_dataset(): #纯 # 使用glob模块查找指定目录下所有的.mat文件，返回这些文件的路径列表
    files = glob.glob('D:/NRSfM/NIPS2022_Yongbo/NRSfM_dataset/**/*.mat')
    return files



if __name__ == '__main__':
#    x = torch.tensor([[1,2,3,4], [1,2,3,4]],requires_grad=True,dtype=torch.float32)
#    b = torch.rand(1, 2)
#    b1 = torch.rand(4, 1)
#    C= torch.matmul(b,x)
#    C1 = torch.matmul(C, b1)
#    C1.backward()
#    x.grad
#    dH = torch.rand(3, 2)
#   测试代码，用于测试PyTorch的梯度计算功能，创建张量、进行矩阵乘法、反向传播计算梯度

    #####################################################################################################
    # Parameters setting for learning
    parser = argparse.ArgumentParser(description='My first deep learning code for NRSfM')  # Input parameters
    parser.add_argument('--batch_size', type=int, default=2,help='Batch size')  # The batch size of the training network
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')  # GPU number
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--all_dataset', type=bool, default=False, help='whether use all dataset for training')

    args = parser.parse_args()  # Add input as parameters
    #####################################################################################################
    # GPU setting
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=config) tensorflow不用了
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #####################################################################################################
    # Load dataset and set the parameters
    #full_result_folder,Scene_normalized,Scene_apoints,J=load_dataset()
    file_names = []
    if args.all_dataset:
        file_names = load_mat_all_dataset()
    else:
        full_result_folder, Scene_normalized, Scene_apoints, J,  file_id= load_mat_dataset()
    #uv=matlab.double(Scene_normalized[[0,1], :].tolist())
    #point_3d= matlab.double(Scene_apoints[[0,1,2], :].tolist())
    #quv=m.fit_python(uv,point_3d,uv,nargout=1)
    #dqu=m.fit_python(uv,point_3d,uv,nargout=2)
    # 注释掉的MATLAB接口调用代码，用于测试拟合函数,将Python数据转换为MATLAB格式并调用MATLAB函数
    #normalized_Image = matlab.double(self.normilized_point_batched[frame_idx, [0, 1], :].tolist())
    # 注释掉的MATLAB数据转换代码，将Python数组转换为MATLAB格式的双精度数组

    points_3D_multiple = []
    y1_ground_multiple = []
    y2_ground_multiple = []
    if file_names:#多文件处理
        for file_id in file_names:
            Scene_normalized, Scene_apoints, J = normalized_points_downsample_load(file_id)
            Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_id, nargout=1))
            points_3D_all, y1_ground, y2_ground = Collect_data(Initial_shape, Scene_normalized, m, device, num_data=10)  # Model parts: shape_partial_derivate and random_depth_data
            points_3D_multiple.append(points_3D_all)
            y1_ground_multiple.append(y1_ground)
            y2_ground_multiple.append(y2_ground)

    else:
        if dataset_params["save_or_load"] == "save":
            #Initial_shape = np.array(m.initialization_for_NRSfM_local_all(nargout=1))
            Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_id[0], nargout=1))
            shape_partial_derivate, random_depth_data = Initial_supervised_learning(Initial_shape, Scene_normalized, m, device, kNN_degree=20, num_iterations=10, num_data=2) # num_iterations=1000, num_data=10 Model parts: shape_partial_derivate and random_depth_data
            #shape_partial_derivate, random_depth_data = Initial_supervised_learning_DGCN(Initial_shape, Scene_normalized, m, kNN_degree=20, num_iterations=2) # Model parts: shape_partial_derivate and random_depth_data
        elif dataset_params["save_or_load"] == "load":
            Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_id[0], nargout=1))
            random_depth_data = []

    PATH = os.path.join(full_result_folder,"0/model.pth")
    PATH1 = os.path.join(full_result_folder, "1/model1.pth")
    try:
        os.mkdir(os.path.join(full_result_folder,"0"))
        os.mkdir(os.path.join(full_result_folder,"1"))
    except OSError: a=1
    else: a=1

    if dataset_params["save_or_load"] == "save":
        torch.save(shape_partial_derivate[0].state_dict(),  PATH)
        torch.save(shape_partial_derivate[1].state_dict(),  PATH1)


    elif dataset_params["save_or_load"] == "load":
        num_point_per_frame = Scene_normalized.shape[1]# 获取每帧的点数
        shape_partial_derivate = []# 初始化模型列表
        num_control_points = num_point_per_frame# 设置控制点数量
        shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=20, mode=0).to(device))
        shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=20, mode=0).to(device))
        shape_partial_derivate[0].load_state_dict(torch.load(PATH))
        shape_partial_derivate[1].load_state_dict(torch.load(PATH1))
        #shape_partial_derivate[0].eval()
        #shape_partial_derivate[1].eval()# 注释掉了模型评估模式设置


    if random_depth_data:
        train_shape_decoder(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints, shape_partial_derivate, random_depth_data, device)
    else:
        train_shape_decoder_GCN(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints, shape_partial_derivate, device)
    #
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
