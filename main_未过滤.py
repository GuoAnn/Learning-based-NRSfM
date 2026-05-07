# This is a sample Python script.
import torch
import os
import matlab.engine
import argparse
import numpy as np
#import scipy.io
import glob
import sys
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from Dataset.dataset_setting import dataset_params
if "DATASET_NAME" in os.environ:
    dataset_params["dataset_name"] = os.environ["DATASET_NAME"]
    
from Dataset.result_setting import result_params
from Dataset.load_dataset import load_preprocessed_W,normalized_points_downsample,normalized_points_without_downsample,normalized_points_downsample_load
from NRSfM_core.train_shape_decoder import train_shape_decoder, train_shape_decoder_GCN
from NRSfM_core.Initial_supervised_learning_DGCN import Initial_supervised_learning_DGCN
#from NRSfM_core.Initial_supervised_learning_framework import Initial_supervised_learning
#from NRSfM_core.Initial_supervised_learning_two_models import Initial_supervised_learning
from NRSfM_core.Initial_supervised_learning_multiple_model import Initial_supervised_learning
from NRSfM_core.Collect_datasets import Collect_data, Initial_learning_from_all_datasets
from NRSfM_core.new_DGCN_model import DGCNNControlPoints


m = matlab.engine.start_matlab()

# === 新增：日志记录器 ===
class Logger(object):
    def __init__(self, filename='log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8') # 追加模式

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_mat_dataset():
    file_path=os.path.join(dataset_params["base_dir"],dataset_params["dataset_name"],"matlab.mat")
    files = glob.glob(os.path.join(dataset_params["base_dir"], dataset_params["dataset_name"], "*.mat"))
    full_result_folder = os.path.join(dataset_params["base_dir"],dataset_params["dataset_name"],"results")
    try:
        os.makedirs(full_result_folder, exist_ok=True)
        print("Successfully created the directory %s " % full_result_folder)
    except OSError:
        print("Creation of the directory %s failed" % full_result_folder)
        
    Scene_normalized, Scene_apoints, J = normalized_points_downsample_load(file_path)
    # 初始化日志
    sys.stdout = Logger(os.path.join(full_result_folder, 'training_log.txt'))
    print(f"✅ Result directory: {full_result_folder}")
    return full_result_folder, Scene_normalized, Scene_apoints, J, files 





if __name__ == '__main__':
#    x = torch.tensor([[1,2,3,4], [1,2,3,4]],requires_grad=True,dtype=torch.float32)
#    b = torch.rand(1, 2)
#    b1 = torch.rand(4, 1)
#    C= torch.matmul(b,x)
#    C1 = torch.matmul(C, b1)
#    C1.backward()
#    x.grad
#    dH = torch.rand(3, 2)

    #####################################################################################################
    # Parameters setting for learning
    parser = argparse.ArgumentParser(description='My first deep learning code for NRSfM')  # Input parameters
    parser.add_argument('--batch_size', type=int, default=2,help='Batch size')  # The batch size of the training network
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')  # GPU number
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--all_dataset', type=bool, default=False, help='Number of epochs')
    # [Added] Resume training from checkpoint
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

    args = parser.parse_args()  # Add input as parameters
    #####################################################################################################
    # GPU setting
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.compat.v1.Session(config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #####################################################################################################
    # Load dataset and set the parameters
    #full_result_folder,Scene_normalized,Scene_apoints,J=load_dataset()
    file_names = []
    full_result_folder, Scene_normalized, Scene_apoints, J,  file_id = load_mat_dataset()
    #uv=matlab.double(Scene_normalized[[0,1], :].tolist())
    #point_3d= matlab.double(Scene_apoints[[0,1,2], :].tolist())
    #quv=m.fit_python(uv,point_3d,uv,nargout=1)
    #dqu=m.fit_python(uv,point_3d,uv,nargout=2)

    #normalized_Image = matlab.double(self.normilized_point_batched[frame_idx, [0, 1], :].tolist())

    points_3D_multiple = []
    y1_ground_multiple = []
    y2_ground_multiple = []
    if file_names:
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
            # [Modified] Added resume and checkpoint_dir parameters
            shape_partial_derivate, random_depth_data = Initial_supervised_learning(
                Initial_shape, Scene_normalized, m, device, kNN_degree=20, 
                num_iterations=1000, num_data=20, 
                resume=args.resume, checkpoint_dir=full_result_folder
            ) 
            # Model parts: shape_partial_derivate and random_depth_data
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
        num_point_per_frame = Scene_normalized.shape[1]
        shape_partial_derivate = []
        num_control_points = num_point_per_frame
        shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=20, mode=0).to(device))
        shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=20, mode=0).to(device))
        shape_partial_derivate[0].load_state_dict(torch.load(PATH))
        shape_partial_derivate[1].load_state_dict(torch.load(PATH1))
        #shape_partial_derivate[0].eval()
        #shape_partial_derivate[1].eval()


    if random_depth_data:
        # [Modified] Added resume parameter
        train_shape_decoder(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints, shape_partial_derivate, random_depth_data, device, resume=args.resume)
    else:
        # [Modified] Added resume parameter
        train_shape_decoder_GCN(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints, shape_partial_derivate, device, resume=args.resume)
    #
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
