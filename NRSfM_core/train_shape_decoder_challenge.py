import numpy as np
import torch
import os
import gc 

from Dataset.load_dataset import get_batched_W
import torch as to
from NRSfM_core.loss_function import NRSfMLoss
from NRSfM_core.shape_decoder import ShapeDecoder, ShapeDecoder_DGNC
from Result_evaluation.Shape_error import shape_error, shape_error_image
from NRSfM_core.GNN_model import Non_LinearGNN

dataset_name = os.environ.get("DATASET_NAME", "")
parts = dataset_name.split("/")
challenge_type = parts[1] if len(parts) > 1 else ""

BASE_GT_DIR = "/home/gax/NRSfM_dataset/challenge_dataset"


# [Modified] Try importing matlab.engine
try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    print("Warning: matlab.engine not installed. Challenge Error will be 0.0")
    HAS_MATLAB = False

def train_shape_decoder(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, model_derivation, device, resume=False):
    # 此函数保持原样，未做修改
    normilized_point_batched,normilized_point_batched_tensor=get_batched_W(normilized_point, device)
    num_frames=normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]
    num_iterations=100000
    kNN_degree=20
    shape_latent_code = to.zeros((num_frames, 1), requires_grad=True, dtype=torch.float32, device=device)
    shape_decoder = ShapeDecoder(num_frames, num_points, Initial_shape, device).to(device)
    shape_decoder = torch.compile(shape_decoder)
    shape_partial_derivate = model_shape
    all_loss_function = NRSfMLoss(normilized_point_batched, num_points, J, m, device, degree=kNN_degree, normilized_point=normilized_point) 
    model_derivation.requires_grad=True
    parameters_to_optimiza = [{'params': shape_latent_code}]
    parameters_to_optimiza.append({"params": shape_partial_derivate[0].parameters()})
    parameters_to_optimiza.append({'params': model_derivation})
    parameters_to_optimiza.append({'params': shape_decoder.parameters()})
    learning_rate = 0.0001
    optimizer = to.optim.Rprop(parameters_to_optimiza, lr=learning_rate, step_sizes=(1e-10, 50))
    error_reported = np.zeros(shape=(1, num_iterations), dtype=np.float32)
    
    start_iter = 0
    ckpt_path = os.path.join(result_folder, "ckpt_network2_mlp_latest.pth") 
    
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming Network 2 (MLP) from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        shape_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        shape_partial_derivate[0].load_state_dict(checkpoint['net1_0_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        with torch.no_grad():
            shape_latent_code.data = checkpoint['latent_code_data']
            model_derivation.data = checkpoint['model_derivation_data']
        start_iter = checkpoint['iteration'] + 1
        print(f"Resumed from iteration {start_iter}")

    try:
        batch_size = 500 
        for i in range(start_iter, num_iterations):
            optimizer.zero_grad()
            cumulative_loss = 0
            for start_f in range(0, num_frames, batch_size):
                end_f = min(start_f + batch_size, num_frames)
                frame_indices = list(range(start_f, end_f))
                loss = all_loss_function.loss_all(shape_decoder, shape_latent_code, shape_partial_derivate, model_derivation, i, frame_indices=frame_indices)
                loss.backward()
                cumulative_loss += loss.item()

            optimizer.step()

            with torch.no_grad():
                depth = shape_decoder.forward(shape_latent_code)
                depth_eval = depth.detach()
            
            normilized_point_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
            for frame_idx in range(depth.shape[0]):
                normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
                normilized_point_result[frame_idx, 2, :] = np.ones(depth.shape[2])
            
            points_3D_result = normilized_point_result * depth_eval.cpu().numpy().repeat(3, 1)

            if Gth is not None and np.any(Gth != 0):
                err_val = shape_error(points_3D_result, Gth, m)
                if isinstance(err_val, torch.Tensor):
                    err_val = err_val.item()
                error_reported[0,i] = err_val
            else:
                error_reported[0,i] = 0.0

            if i % 10 == 0:  
                print('[%5d, %5d] loss: %.3f accuracy: %.6f' %(i + 1, num_iterations, cumulative_loss, error_reported[0,i]))
            
            if (i + 1) % 3 == 0:
                print(f"Saving Checkpoint at iteration {i+1}...")
                torch.save({
                    'iteration': i,
                    'decoder_state_dict': shape_decoder.state_dict(),
                    'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                    'latent_code_data': shape_latent_code.data,
                    'model_derivation_data': model_derivation.data,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                torch.save(depth, os.path.join(result_folder, "depth_latest.pt"))
                del depth, depth_eval, points_3D_result
                gc.collect()

    except Exception as e:
        print(f"\nNetwork 2 (MLP) training interrupted at iteration {i} due to error: {e}")
        try:
            torch.save({
                'iteration': i,
                'decoder_state_dict': shape_decoder.state_dict(),
                'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                'latent_code_data': shape_latent_code.data,
                'model_derivation_data': model_derivation.data,
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
        except:
            pass
        raise e

    print('\n\n\n', 'Compiling complete')
    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code, os.path.join(result_folder, "shape_latent_code.pt"))
    torch.save(depth, os.path.join(result_folder, "depth.pt"))
    
    if Gth is not None and np.any(Gth != 0):
        depth_final = shape_decoder.forward(shape_latent_code).detach()
        points_3D_final = normilized_point_result * depth_final.cpu().numpy().repeat(3, 1)
        error_reported[0, i] = shape_error_image(points_3D_final, Gth, m)

    return 1

# [Modified] Added resume parameter and Debugging
def train_shape_decoder_GCN(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, device, resume=False):
    normilized_point_batched,normilized_point_batched_tensor=get_batched_W(normilized_point, device)
    num_frames=normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]
    num_iterations=200
    kNN_degree=20
    shape_latent_code = to.zeros((num_frames, 1), requires_grad=True, dtype=torch.float32, device=device)
    network_model = "MLP"
    if network_model== "MLP":
        shape_decoder = ShapeDecoder(num_frames, num_points, Initial_shape, device).to(device)
    elif network_model== "DGNC":
        shape_decoder = ShapeDecoder_DGNC(num_points, num_points=20, mode=0).to(device)
    shape_partial_derivate = model_shape
    all_loss_function = NRSfMLoss(normilized_point_batched, num_points, J, m, device, degree=kNN_degree, normilized_point=normilized_point) 
    if network_model == "MLP":
        parameters_to_optimiza = [{'params': shape_latent_code}]
    elif network_model == "DGNC":
        parameters_to_optimiza = []
        shape_latent_code = to.zeros((num_frames, 3, num_points), requires_grad=False, dtype=torch.float32, device=device)
        shape_latent_code[:, [0,1], :] = normilized_point_batched_tensor
    
    parameters_to_optimiza.append({'params': shape_decoder.parameters()})
    learning_rate = 0.00001
    optimizer = to.optim.Rprop(parameters_to_optimiza, lr=learning_rate, step_sizes=(1e-10, 0.01))
    #optimizer = torch.optim.Adam(parameters_to_optimiza, lr=learning_rate)
    error_reported = np.zeros(shape=(1, num_iterations), dtype=np.float32)
    
    start_iter = 0
    ckpt_path = os.path.join(result_folder, "ckpt_network2_latest.pth")
    
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming Network 2 (GCN) from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        shape_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        shape_partial_derivate[0].load_state_dict(checkpoint['net1_0_state_dict'])
        shape_partial_derivate[1].load_state_dict(checkpoint['net1_1_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        with torch.no_grad():
            shape_latent_code.data = checkpoint['latent_code_data']
        start_iter = checkpoint['iteration'] + 1
        print(f"Resumed from iteration {start_iter}")

    # =========================================================================
    # [DEBUG START] 详细的初始化检查
    # =========================================================================
    print("\n" + "="*50)
    print("Initializing Evaluation Module (FIXED MODE)...")
    
    # 1. 检查 GT 文件路径
    # [请根据你的实际情况修改这里]
    # 注意：你之前的日志报错说找不到 /articulated/gt_frame_25.txt
    # 请确认文件名是否真的是 gt_frame_25.txt，有没有大小写或下划线错误？
    GT_CONFIG = {
    "articulated": ("gt_frame_103.txt", 102),
    "balloon_vis": ("gt_frame_25.txt", 24),
    "paper_vis": ("gt_frame_20.txt", 19),
    "stretch": ("gt_frame_20.txt", 19),
    "tearing_vis": ("gt_frame_202.txt", 201),
}

    if challenge_type not in GT_CONFIG:
        raise ValueError(f"Unknown challenge type: {challenge_type}")

    gt_file, challenge_frame_idx = GT_CONFIG[challenge_type]
    challenge_gt_path = os.path.join(BASE_GT_DIR, challenge_type, gt_file) 
    challenge_gt_data = None

    if os.path.exists(challenge_gt_path):
        try:
            challenge_gt_data = np.loadtxt(challenge_gt_path)
            if challenge_gt_data.shape[0] != 3:
                challenge_gt_data = challenge_gt_data.T
            print(f"[DEBUG] Challenge GT loaded. Shape: {challenge_gt_data.shape}")
        except Exception as e:
            print(f"[DEBUG] Error loading GT file: {e}")
            challenge_gt_data = None
    else:
        print(f"[DEBUG] ❌ GT file NOT FOUND at: {challenge_gt_path}")
        print("[DEBUG] WARNING: Ch_Err will be 0.0. Please check the path carefully!")

    # 2. 启动 Matlab 引擎 & 智能搜索路径
    eng = None
    if HAS_MATLAB:
        try:
            print("[DEBUG] Starting Matlab Engine...")
            eng = matlab.engine.start_matlab()
            
            # 获取当前脚本所在目录
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 尝试多个可能的路径寻找 Result_evaluation
            possible_paths = [
                os.path.join(current_script_dir, 'Result_evaluation'),           # 子目录
                os.path.join(os.path.dirname(current_script_dir), 'Result_evaluation'), # 兄弟目录 (通常是这个)
                os.path.join(os.getcwd(), 'Result_evaluation')                   # 工作目录下的子目录
            ]
            
            final_matlab_path = None
            for p in possible_paths:
                if os.path.exists(p) and os.path.exists(os.path.join(p, 'nrsfm_score.m')):
                    final_matlab_path = p
                    break
            
            if final_matlab_path:
                print(f"[DEBUG] Found Matlab scripts at: {final_matlab_path}")
                eng.addpath(final_matlab_path)
                
                # Double check
                if int(eng.exist('nrsfm_score', nargout=1)) != 0:
                    print("[DEBUG] ✅ Matlab Engine ready.")
                else:
                    print("[DEBUG] ❌ Added path but cannot find function. Something is weird.")
            else:
                print("[DEBUG] ❌ Could not find 'Result_evaluation' folder containing 'nrsfm_score.m'")
                print(f"[DEBUG] Searched locations: {possible_paths}")
                
        except Exception as e:
            print(f"[DEBUG] Failed to start Matlab Engine: {e}")
            eng = None
    else:
        print("[DEBUG] matlab.engine package is not installed.")

    print("="*50 + "\n")
    # =========================================================================

    try:
        batch_size = 500
        for i in range(start_iter, num_iterations):
            shape_partial_derivate[0].train()
            shape_partial_derivate[1].train()
            shape_decoder.train()
            optimizer.zero_grad()
            
            cumulative_loss = 0
            for start_f in range(0, num_frames, batch_size):
                end_f = min(start_f + batch_size, num_frames)
                frame_indices = list(range(start_f, end_f))
                
                loss = all_loss_function.loss_all_GNC(shape_decoder, shape_latent_code, shape_partial_derivate, i, network_model, torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device), frame_indices=frame_indices)
                loss.backward()
                cumulative_loss += loss.item()

            optimizer.step()
            
            shape_partial_derivate[0].eval()
            shape_partial_derivate[1].eval()
            shape_decoder.eval()

            with torch.no_grad():
                depth = shape_decoder.forward(shape_latent_code)
            
            if network_model == "MLP":
                depth_eval = depth.detach()
            elif network_model == "DGNC":
                depth_eval = depth.detach() + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
                depth_eval = torch.unsqueeze(depth_eval, 1)
            
            normilized_point_result = np.zeros(shape=(depth_eval.shape[0], 3, depth_eval.shape[2]), dtype=np.float32)
            for frame_idx in range(depth_eval.shape[0]):
                normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
                normilized_point_result[frame_idx, 2, :] = np.ones(depth_eval.shape[2])
            
            points_3D_result = normilized_point_result * depth_eval.cpu().numpy().repeat(3, 1)

            if Gth is not None and np.any(Gth != 0):
                err_val = shape_error(points_3D_result, Gth, m)
                if isinstance(err_val, torch.Tensor):
                    err_val = err_val.item()
                error_reported[0,i] = err_val
            else:
                error_reported[0,i] = 0.0

            # [Modified] Matlab Engine Evaluation
            challenge_error_val = 0.0
            
            if eng is not None and challenge_gt_data is not None:
                # 每帧都算
                if i % 5 == 0: 
                    try:
                        if challenge_frame_idx < points_3D_result.shape[0]:
                            pred_frame = points_3D_result[challenge_frame_idx, :, :] 
                            pred_list = pred_frame.tolist()
                            gt_list = challenge_gt_data.tolist()
                            
                            pred_mat = matlab.double(pred_list)
                            gt_mat = matlab.double(gt_list)
                            
                            ret = eng.nrsfm_score(pred_mat, gt_mat, nargout=3)
                            challenge_error_val = ret[0]
                    except Exception as e:
                        # 第一次报错打印，后续如果太频繁可以注释掉
                        if i == start_iter:
                            print(f"[Error] Matlab execution failed: {e}")
            elif i == start_iter:
                # 再次提醒
                 print(f"[DEBUG] Skipping Matlab Eval because initialization failed.")

            if i % 5 == 0:
                print('[%5d, %5d] loss: %.3f accuracy: %.6f | Ch_Err: %.6f' % (
                    i + 1, num_iterations, cumulative_loss, error_reported[0,i], challenge_error_val))
           
            if  (i + 1) % 100 == 0: 
                depth_filename = f"depth_{i}.pt"
                depth_save_path = os.path.join(result_folder, depth_filename)
                torch.save(depth, depth_save_path)
                print(f"Saved depth file for iteration {i} to {depth_save_path}")
            
            if (i + 1) % 100 == 0:
                print(f"Saving Checkpoint at iteration {i+1}...")
                torch.save({
                    'iteration': i,
                    'decoder_state_dict': shape_decoder.state_dict(),
                    'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                    'net1_1_state_dict': shape_partial_derivate[1].state_dict(),
                    'latent_code_data': shape_latent_code.data, 
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                torch.save(depth, os.path.join(result_folder, "depth_latest.pt"))
                del depth, depth_eval, points_3D_result
                gc.collect()

    except Exception as e:
        print(f"\nNetwork 2 (GCN) training interrupted at iteration {i} due to error: {e}")
        try:
            torch.save({
                'iteration': i,
                'decoder_state_dict': shape_decoder.state_dict(),
                'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                'net1_1_state_dict': shape_partial_derivate[1].state_dict(),
                'latent_code_data': shape_latent_code.data,
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
        except:
            pass
        if eng is not None:
            eng.quit()
        raise e

    print('\n\n\n', 'Compiling complete')
    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code, os.path.join(result_folder, "shape_latent_code.pt"))
    
    with torch.no_grad():
        depth_final = shape_decoder.forward(shape_latent_code)
        if network_model == "DGNC":
            depth_final = depth_final + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
            depth_final = torch.unsqueeze(depth_final, 1)

    torch.save(depth_final, os.path.join(result_folder, "depth_new.pt"))

    if Gth is not None and np.any(Gth != 0):
        depth_final = shape_decoder.forward(shape_latent_code).detach()
        if network_model == "DGNC":
             depth_final = depth_final + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
             depth_final = torch.unsqueeze(depth_final, 1)
        
        points_3D_final = normilized_point_result * depth_final.cpu().numpy().repeat(3, 1)
        error_reported[0, i] = shape_error_image(points_3D_final, Gth, m)
        
    final_eval_error = error_reported[0, i]
    
    challenge_final_msg = ""
    if challenge_gt_data is not None:
         challenge_final_msg = f" | Final Challenge Error: {challenge_error_val:.6f}"
         
    print(f"\nFinal Evaluation Accuracy (shape_error_image): {final_eval_error:.6f}{challenge_final_msg}")

    if eng is not None:
        eng.quit()

    return 1