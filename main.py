import torch
import os
import matlab.engine
import argparse
import numpy as np
import scipy.io as sio
import glob
import sys

# Press Shift+F10 to execute it or replace it with your code. #main for tricky
from Dataset.dataset_setting import dataset_params
if "DATASET_NAME" in os.environ:
    dataset_params["dataset_name"] = os.environ["DATASET_NAME"]
    
from Dataset.result_setting import result_params
from Dataset.load_dataset import load_preprocessed_W,normalized_points_downsample,normalized_points_without_downsample,normalized_points_downsample_load
from NRSfM_core.train_shape_decoder import train_shape_decoder, train_shape_decoder_GCN
from NRSfM_core.Initial_supervised_learning_DGCN import Initial_supervised_learning_DGCN
from NRSfM_core.Initial_supervised_learning_multiple_model import Initial_supervised_learning
from NRSfM_core.Collect_datasets import Collect_data, Initial_learning_from_all_datasets
from NRSfM_core.new_DGCN_model import DGCNNControlPoints


m = matlab.engine.start_matlab()

# === 新增：日志记录器 ===
class Logger(object):
    def __init__(self, filename='log.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

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


# === 递归注入噪声函数 (核心修复) ===
def recursive_inject_noise(data, key_path, noise_scale=1e-6):
    """
    递归遍历 scipy.io 加载的复杂结构，给所有大的浮点矩阵加噪声。
    """
    injected_count = 0
    
    # 情况1: 直接是 numpy 数组
    if isinstance(data, np.ndarray):
        # 如果是浮点数且尺寸够大 -> 加噪声
        if data.dtype.kind in 'fc' and data.size > 20: # 阈值调低到20以防万一
            noise = np.random.normal(0, noise_scale, data.shape)
            data += noise
            print(f"   -> [INJECTED] {key_path} (Shape: {data.shape}, Type: Float)")
            return 1
        
        # 情况2: Matlab Struct (NumPy Structured Array)
        # scipy.io 加载 struct 时，dtype.names 不为空
        if data.dtype.names is not None:
            # print(f"   -> [Recurse Struct] {key_path}")
            for field in data.dtype.names:
                # 递归处理每个字段
                # 注意：struct 里的字段通常还是包了一层 array，比如 data[field][0,0]
                # 我们直接传 data[field] 让递归逻辑去解
                injected_count += recursive_inject_noise(data[field], f"{key_path}.{field}", noise_scale)
            return injected_count

        # 情况3: Object Array (通常是 Matlab Cell Array 或 Struct 的容器)
        if data.dtype.kind == 'O':
            # print(f"   -> [Recurse Object] {key_path}")
            # 遍历对象数组中的每个元素
            # 使用 np.nditer 可能只读，这里用 flat 迭代器
            flat_iter = data.flat
            for i in range(data.size):
                element = flat_iter[i]
                # 递归修改元素
                # 注意：如果是基本类型，这里修改 element 可能不会回写到 data[i]
                # 只有当 element 是引用类型（如 array, object）时才有效
                # 对于 scipy.io 加载的复杂结构，通常是嵌套的 array，所以是引用
                injected_count += recursive_inject_noise(element, f"{key_path}[{i}]", noise_scale)
            return injected_count

    return injected_count


if __name__ == '__main__':
    #####################################################################################################
    # Parameters setting for learning
    parser = argparse.ArgumentParser(description='My first deep learning code for NRSfM')
    parser.add_argument('--batch_size', type=int, default=2,help='Batch size')
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--all_dataset', type=bool, default=False, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #####################################################################################################
    # Load dataset
    file_names = []
    full_result_folder, Scene_normalized, Scene_apoints, J,  file_id = load_mat_dataset()

    points_3D_multiple = []
    y1_ground_multiple = []
    y2_ground_multiple = []
    
    if file_names:
        for file_id in file_names:
            Scene_normalized, Scene_apoints, J = normalized_points_downsample_load(file_id)
            Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_id, nargout=1))
            points_3D_all, y1_ground, y2_ground = Collect_data(Initial_shape, Scene_normalized, m, device, num_data=10)
            points_3D_multiple.append(points_3D_all)
            y1_ground_multiple.append(y1_ground)
            y2_ground_multiple.append(y2_ground)

    else:
        if dataset_params["save_or_load"] == "save":
            
            # ================= [Added] 数据质量诊断与强力修复模块 START =================
            print(f"\n[DIAGNOSIS] Checking data quality for: {file_id[0]}")
            
            # 1. 检查 NaN/Inf (内存中)
            if np.isnan(Scene_normalized).any() or np.isinf(Scene_normalized).any():
                print("❌ CRITICAL: Input data contains NaN or Inf values!")
            else:
                print("✅ Data check passed: No static points found (in memory).")

            # ★★★ 强力修复模式：递归注入噪声 ★★★
            print("🔧 Applying tiny Gaussian noise (jitter) to stabilize MATLAB initialization...")
            noise_scale = 1e-6 
            file_to_load = file_id[0]

            try:
                # 读取原始 mat (不进行 simplify_cells，保持结构以便修改)
                mat_data = sio.loadmat(file_id[0])
                print(f"   [DEBUG] Top-level keys: {list(mat_data.keys())}") 

                total_injected = 0
                keys = list(mat_data.keys())
                for key in keys:
                    if key.startswith('__'): continue
                    
                    # 调用递归函数处理每一个变量
                    # scipy.io 读取的 struct 往往是 object array 或者是 structured array
                    # 这个函数会钻进去找到里面的 float 矩阵并加噪声
                    total_injected += recursive_inject_noise(mat_data[key], key, noise_scale)

                if total_injected > 0:
                    temp_mat_path = os.path.join(full_result_folder, "temp_jittered.mat")
                    sio.savemat(temp_mat_path, mat_data)
                    print(f"✅ Modified {total_injected} matrices. Saved to: {temp_mat_path}")
                    file_to_load = temp_mat_path
                else:
                    print("⚠️ WARNING: Recursive search found NO valid float matrices to inject noise.")
                    print("   Please check if the .mat file format is extremely unusual.")

            except Exception as e:
                print(f"⚠️ Error injecting noise: {e}. Using original file.")
                import traceback
                traceback.print_exc()

            print("============================================================\n")
            # ================= [Added] 数据质量诊断模块 END ===================

            # 调用 MATLAB，传入修复后的文件
            Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_to_load, nargout=1))
            
            # 保持内存一致性
            if file_to_load != file_id[0]:
                 Scene_normalized += np.random.normal(0, noise_scale, Scene_normalized.shape)

            shape_partial_derivate, random_depth_data = Initial_supervised_learning(
                Initial_shape, Scene_normalized, m, device, kNN_degree=20, 
                num_iterations=10, num_data=20, 
                resume=args.resume, checkpoint_dir=full_result_folder
            ) 
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


    if random_depth_data:
        train_shape_decoder(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints, shape_partial_derivate, random_depth_data, device, resume=args.resume)
    else:
        train_shape_decoder_GCN(full_result_folder, Scene_normalized, args, J, m, Initial_shape, Scene_apoints, shape_partial_derivate, device, resume=args.resume)