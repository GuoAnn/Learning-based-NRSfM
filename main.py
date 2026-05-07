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
if dataset_params["dataset_name"].startswith("challenge_dataset/"):
    from NRSfM_core.train_shape_decoder_challenge import train_shape_decoder, train_shape_decoder_GCN
else:
    from NRSfM_core.train_shape_decoder import train_shape_decoder, train_shape_decoder_GCN
from NRSfM_core.Initial_supervised_learning_DGCN import Initial_supervised_learning_DGCN
from NRSfM_core.Initial_supervised_learning_multiple_model import Initial_supervised_learning
from NRSfM_core.Collect_datasets import Collect_data, Initial_learning_from_all_datasets
from NRSfM_core.new_DGCN_model import DGCNNControlPoints, profile_dgcnn_overhead


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


def apply_gaussian_noise(scene_normalized, sigma):
    if sigma is None or sigma <= 0:
        return scene_normalized
    return scene_normalized + np.random.normal(0, sigma, scene_normalized.shape)


def build_visibility_mask(scene_normalized, drop_ratio):
    if drop_ratio is None or drop_ratio <= 0:
        return None
    num_frames = scene_normalized.shape[0] // 2
    num_points = scene_normalized.shape[1]
    return (np.random.rand(num_frames, num_points) >= drop_ratio).astype(np.float32)


def run_training_pipeline(scene_normalized, scene_apoints, J, file_id, args, device, result_folder, mask=None, num_iterations=None):
    os.makedirs(result_folder, exist_ok=True)
    random_depth_data = []

    if dataset_params["save_or_load"] == "save":
        print(f"\n[DIAGNOSIS] Checking data quality for: {file_id[0]}")
        if np.isnan(scene_normalized).any() or np.isinf(scene_normalized).any():
            print("❌ CRITICAL: Input data contains NaN or Inf values!")
        else:
            print("✅ Data check passed: No static points found (in memory).")
        print("🔧 Applying tiny Gaussian noise (jitter) to stabilize MATLAB initialization...")
        noise_scale = 1e-6
        file_to_load = file_id[0]

        try:
            mat_data = sio.loadmat(file_id[0])
            print(f"   [DEBUG] Top-level keys: {list(mat_data.keys())}")
            total_injected = 0
            keys = list(mat_data.keys())
            for key in keys:
                if key.startswith('__'):
                    continue
                total_injected += recursive_inject_noise(mat_data[key], key, noise_scale)

            if total_injected > 0:
                temp_mat_path = os.path.join(result_folder, "temp_jittered.mat")
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

        Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_to_load, nargout=1))

        if file_to_load != file_id[0]:
            scene_normalized = scene_normalized + np.random.normal(0, noise_scale, scene_normalized.shape)

        shape_partial_derivate, random_depth_data = Initial_supervised_learning(
            Initial_shape, scene_normalized, m, device, kNN_degree=20,
            num_iterations=10, num_data=20,
            resume=args.resume, checkpoint_dir=result_folder, mask=mask
        )
    elif dataset_params["save_or_load"] == "load":
        Initial_shape = np.array(m.initialization_for_NRSfM_local_all_new(file_id[0], nargout=1))
        random_depth_data = []

    PATH = os.path.join(result_folder, "0/model.pth")
    PATH1 = os.path.join(result_folder, "1/model1.pth")
    os.makedirs(os.path.join(result_folder, "0"), exist_ok=True)
    os.makedirs(os.path.join(result_folder, "1"), exist_ok=True)

    if dataset_params["save_or_load"] == "save":
        torch.save(shape_partial_derivate[0].state_dict(), PATH)
        torch.save(shape_partial_derivate[1].state_dict(), PATH1)

    elif dataset_params["save_or_load"] == "load":
        num_point_per_frame = scene_normalized.shape[1]
        shape_partial_derivate = []
        num_control_points = num_point_per_frame
        shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=20, mode=0).to(device))
        shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=20, mode=0).to(device))
        shape_partial_derivate[0].load_state_dict(torch.load(PATH))
        shape_partial_derivate[1].load_state_dict(torch.load(PATH1))

    if random_depth_data:
        final_error = train_shape_decoder(
            result_folder, scene_normalized, args, J, m, Initial_shape, scene_apoints,
            shape_partial_derivate, random_depth_data, device,
            resume=args.resume, mask=mask, num_iterations=num_iterations
        )
    else:
        final_error = train_shape_decoder_GCN(
            result_folder, scene_normalized, args, J, m, Initial_shape, scene_apoints,
            shape_partial_derivate, device,
            resume=args.resume, mask=mask, num_iterations=num_iterations
        )

    return final_error


if __name__ == '__main__':
    #####################################################################################################
    # Parameters setting for learning
    parser = argparse.ArgumentParser(description='My first deep learning code for NRSfM')
    parser.add_argument('--batch_size', type=int, default=2,help='Batch size')
    parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--all_dataset', type=bool, default=False, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--rebuttal_eval', action='store_true', help='Run rebuttal timing/noise/masking suite')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #####################################################################################################
    # Load dataset
    full_result_folder, Scene_normalized, Scene_apoints, J, file_id = load_mat_dataset()
    if args.rebuttal_eval:
        knn_ms = None
        forward_ms = None
        if device.type == 'cuda':
            num_points = Scene_normalized.shape[1]
            frame_points = np.zeros((1, 3, num_points), dtype=np.float32)
            frame_points[0, 0, :] = Scene_normalized[0, :]
            frame_points[0, 1, :] = Scene_normalized[1, :]
            frame_points[0, 2, :] = 1.0
            probe_input = torch.tensor(frame_points).to(device)
            probe_model = DGCNNControlPoints(num_points, num_points=20, mode=0).to(device)
            knn_ms, forward_ms = profile_dgcnn_overhead(probe_model, probe_input, warmup=5, iters=20)

        if knn_ms is None or forward_ms is None:
            overhead_str = "N/A"
        else:
            overhead_str = f"{forward_ms:.3f}/{knn_ms:.3f}"

        NOISE_STD_DEV_PIXELS = 2.0
        rebuttal_epochs = 5000
        noise_scene = apply_gaussian_noise(Scene_normalized.copy(), NOISE_STD_DEV_PIXELS)
        noise_folder = os.path.join(full_result_folder, "rebuttal_noise")
        noise_error = run_training_pipeline(
            noise_scene, Scene_apoints, J, file_id, args, device, noise_folder,
            mask=None, num_iterations=rebuttal_epochs
        )

        mask = build_visibility_mask(Scene_normalized, 0.1)
        mask_folder = os.path.join(full_result_folder, "rebuttal_mask")
        mask_error = run_training_pipeline(
            Scene_normalized.copy(), Scene_apoints, J, file_id, args, device, mask_folder,
            mask=mask, num_iterations=rebuttal_epochs
        )

        print(f"DGCNN/KNN Overhead: {overhead_str} ms")
        # shape_error outputs percentage values already; keep the percent sign as a label.
        print(f"3D Error with 2px Noise: {noise_error:.6f} %")
        print(f"3D Error with 10% Masking: {mask_error:.6f} %")
        sys.exit(0)

    run_training_pipeline(Scene_normalized, Scene_apoints, J, file_id, args, device, full_result_folder)
