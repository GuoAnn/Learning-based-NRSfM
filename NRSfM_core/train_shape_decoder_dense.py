import numpy as np
import torch
import os
import gc # [Added] for memory cleaning

from Dataset.load_dataset import get_batched_W
import torch as to
from NRSfM_core.loss_function import NRSfMLoss
#from NRSfM_core.model_develop import learning_model
from NRSfM_core.shape_decoder import ShapeDecoder, ShapeDecoder_DGNC
from Result_evaluation.Shape_error import shape_error, shape_error_image
from NRSfM_core.GNN_model import Non_LinearGNN

# ★ [Added] 修复后的 e3D 计算函数，参考 File 1 ★
# 参考论文 [Sidhu2020] 的定义：e3D = 1/T Σ_t ||S_t^GT - S_t||_F / ||S_t^GT||_F
'''def compute_dense_e3d(prediction, ground_truth, do_scale=True, outlier_threshold=0.5):
    """
    计算 dense e3D error，逻辑严格对齐 MATLAB 代码。
    
    prediction: (F, 3, P) 重建结果
    ground_truth: (F, 3, P) GT
    outlier_threshold: 离群点过滤阈值。
                       注意：如果数据是归一化的(例如在[-1,1])，阈值可能是 0.05 或 0.1。
                       如果数据是世界坐标(毫米)，阈值可能是 50 或 100。
                       如果不传(None)，则不进行二次过滤（对应 MATLAB 注释掉 idx1 判断的情况）。
    """
    F, _, P = prediction.shape
    error_sum = 0.0
    valid_frames = 0
    
    for t in range(F):
        P_pred = prediction[t]   # (3, P)
        P_gt = ground_truth[t]   # (3, P)
        
        # 1. MATLAB 逻辑: 找出非零的有效点 (idx = find(P1o(:,1)~=0))
        # 我们假设任意坐标不为0即为有效，或者范数不为0
        valid_mask = np.sum(np.abs(P_pred), axis=0) > 1e-6
        if np.sum(valid_mask) < 3: # 点太少无法计算 Procrustes
            continue
            
        # 只取有效点进行第一次对齐
        A = P_pred[:, valid_mask]
        B = P_gt[:, valid_mask]
        
        # Center
        mu_A = A.mean(axis=1, keepdims=True)
        mu_B = B.mean(axis=1, keepdims=True)
        A_centered = A - mu_A
        B_centered = B - mu_B
        
        # Procrustes 1: find R, s
        H = B_centered @ A_centered.T
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
            
        A_rot = R @ A_centered
        if do_scale:
            # s = sum(trace(A_rot * B)) / sum(trace(A_rot * A_rot))
            s = np.sum(A_rot * B_centered) / (np.sum(A_rot * A_rot) + 1e-10)
        else:
            s = 1.0
            
        aligned_1 = s * A_rot + mu_B # 第一次对齐后的 A
        
        # 2. MATLAB 逻辑: 二次过滤 (Thresholding)
        # idx1 = find(norm(P1_old - P) < threshold)
        final_mask = np.ones(A.shape[1], dtype=bool) # 在 valid_mask 基础上的 mask
        
        if outlier_threshold is not None:
            # 计算欧氏距离
            dists = np.sqrt(np.sum((B - aligned_1) ** 2, axis=0))
            inlier_sub_mask = dists < outlier_threshold
            
            # 如果内点太少，就跳过过滤步骤，保留第一次的结果(防止报错)
            if np.sum(inlier_sub_mask) > 3:
                final_mask = inlier_sub_mask
                
                # 准备第二次 Procrustes 的数据
                A_2 = A[:, final_mask]
                B_2 = B[:, final_mask]
                
                # Center 2
                mu_A2 = A_2.mean(axis=1, keepdims=True)
                mu_B2 = B_2.mean(axis=1, keepdims=True)
                A2_centered = A_2 - mu_A2
                B2_centered = B_2 - mu_B2
                
                # Procrustes 2
                H2 = B2_centered @ A2_centered.T
                U2, S2, Vt2 = np.linalg.svd(H2)
                R2 = U2 @ Vt2
                if np.linalg.det(R2) < 0:
                    Vt2[-1, :] *= -1
                    R2 = U2 @ Vt2
                
                A2_rot = R2 @ A2_centered
                if do_scale:
                    s2 = np.sum(A2_rot * B2_centered) / (np.sum(A2_rot * A2_rot) + 1e-10)
                else:
                    s2 = 1.0
                    
                aligned_final = s2 * A2_rot
                B_final_centered = B2_centered # Error calculation uses centered data usually or aligned absolute
                # MATLAB Code: P1x(idx2,:) - P(idx2,:)
                # MATLAB 的误差计算是在对齐后的绝对坐标上算的，分子分母同理
                
                # 为了严格匹配 MATLAB: 
                # sc(i) = sqrt(sum(sum(P(idx2,:).^2))) -> GT 的 Frobenius Norm
                # ep1(i) = sqrt(sum(sum((P1x - P).^2))) / sc(i)
                
                # 重新计算绝对坐标的对齐结果
                # Prediction aligned to GT: s2 * R2 * (A_2 - mu_A2) + mu_B2
                P_pred_final = s2 * (R2 @ (A_2 - mu_A2)) + mu_B2
                P_gt_final = B_2
                
            else:
                # 没过滤成功，沿用第一次
                P_pred_final = aligned_1
                P_gt_final = B
        else:
            # 没有阈值，沿用第一次
            P_pred_final = aligned_1
            P_gt_final = B

        # 3. 最终误差计算
        # Numerator: || Pred_aligned - GT ||_F
        diff_norm = np.linalg.norm(P_pred_final - P_gt_final, 'fro')
        # Denominator: || GT ||_F
        gt_norm = np.linalg.norm(P_gt_final, 'fro')
        
        error_sum += diff_norm / (gt_norm + 1e-10)
        valid_frames += 1
    
    return error_sum / max(valid_frames, 1)'''

import numpy as np

def compute_dense_e3d(prediction, ground_truth, outlier_threshold=70.0):
    """
    [Final Debug Version] 计算 Dense e3D。
    outlier_threshold: Paper数据集设为70.0, Tshirt数据集设为50.0
    """
    # 1. 转换 Tensor -> Numpy
    if hasattr(prediction, 'cpu'): prediction = prediction.detach().cpu().numpy()
    if hasattr(ground_truth, 'cpu'): ground_truth = ground_truth.detach().cpu().numpy()
    if np.isnan(prediction).any(): prediction = np.nan_to_num(prediction)

    # 2. 打印诊断信息 (只打印一次或出错时打印)
    p_max = np.max(prediction)
    # 如果预测值太小(说明还没训练起来)，就不打印满屏的Debug了，但依然计算
    if p_max > 1e-5: 
        print(f"  [e3D Check] Pred Max: {p_max:.2f}, GT Shape: {ground_truth.shape}")

    F, _, N = prediction.shape
    error_sum = 0.0
    valid_frames = 0
    
    for t in range(F):
        P_pred = prediction[t]
        P_gt = ground_truth[t]
        
        # --- 有效性检查: 只看 GT 是否有值 ---
        valid_mask = np.sum(np.abs(P_gt), axis=0) > 1e-6
        if np.sum(valid_mask) < 10: continue

        A = P_pred[:, valid_mask]
        B = P_gt[:, valid_mask]
        
        # --- Procrustes 对齐 ---
        mu_A = A.mean(axis=1, keepdims=True)
        mu_B = B.mean(axis=1, keepdims=True)
        A_centered = A - mu_A
        B_centered = B - mu_B
        
        H = B_centered @ A_centered.T
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = U @ Vt
            
        A_rot = R @ A_centered
        denom = np.sum(A_rot * A_rot)
        s = np.sum(A_rot * B_centered) / (denom + 1e-10) # 自动计算 Scale
        
        aligned_1 = s * A_rot + mu_B 
        
        # --- 离群点过滤 ---
        final_mask = np.ones(A.shape[1], dtype=bool)
        if outlier_threshold is not None:
            dists = np.sqrt(np.sum((B - aligned_1) ** 2, axis=0))
            inlier_sub_mask = dists < outlier_threshold
            
            if np.sum(inlier_sub_mask) > 10:
                final_mask = inlier_sub_mask
                # 二次对齐
                A_2 = A[:, final_mask]
                B_2 = B[:, final_mask]
                mu_A2 = A_2.mean(axis=1, keepdims=True)
                mu_B2 = B_2.mean(axis=1, keepdims=True)
                A2_c = A_2 - mu_A2
                B2_c = B_2 - mu_B2
                H2 = B2_c @ A2_c.T
                U2, S2, Vt2 = np.linalg.svd(H2)
                R2 = U2 @ Vt2
                if np.linalg.det(R2) < 0: Vt2[-1, :] *= -1; R2 = U2 @ Vt2
                A2_rot = R2 @ A2_c
                denom2 = np.sum(A2_rot * A2_rot)
                s2 = np.sum(A2_rot * B2_c) / (denom2 + 1e-10)
                P_pred_final = s2 * (R2 @ (A_2 - mu_A2)) + mu_B2
                P_gt_final = B_2
            else:
                P_pred_final = aligned_1
                P_gt_final = B
        else:
            P_pred_final = aligned_1
            P_gt_final = B

        # --- 误差计算 ---
        diff_norm = np.linalg.norm(P_pred_final - P_gt_final, 'fro')
        gt_norm = np.linalg.norm(P_gt_final, 'fro')
        error_sum += diff_norm / (gt_norm + 1e-10)
        valid_frames += 1

    return error_sum / max(valid_frames, 1)


# [Modified] Added resume parameter
def train_shape_decoder(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, model_derivation, device, resume=False):
    normilized_point_batched,normilized_point_batched_tensor=get_batched_W(normilized_point, device)
    num_frames=normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]
    num_iterations=100000
    kNN_degree=20
    ## Tensorfolow
    #shape_latent_code = tf.random.normal([num_frames,1], 0, 1, tf.float32, seed=1)
    ## Pytorch
    #shape_latent_code = to.randn((num_frames,1),requires_grad=True, dtype=torch.float32)
    shape_latent_code = to.zeros((num_frames, 1), requires_grad=True, dtype=torch.float32, device=device)
    #May change into num_points parameter: num_points=normilized_point_batched.shape[1]
    shape_decoder = ShapeDecoder(num_frames, num_points, Initial_shape, device).to(device)

    shape_decoder = torch.compile(shape_decoder)#新加的
    #for p in shape_decoder.parameters():
    #    p.data.fill_(1)
    #shape_partial_derivate = Non_LinearGNN(num_points, feat_dim=3, stat_dim=3, iteration=5, degree=kNN_degree)
    shape_partial_derivate = model_shape
    ################################ Learning Model Initialization################################
    all_loss_function = NRSfMLoss(normilized_point_batched, num_points, J, m, device, degree=kNN_degree, normilized_point=normilized_point) # degree is the
    ################################ Initial loss################################
    #loss_f0 = all_loss_function.loss_all(shape_decoder, shape_latent_code, shape_partial_derivate, model_derivation)
    ######################### Trainning ################################
    model_derivation.requires_grad=True
    parameters_to_optimiza = [{'params': shape_latent_code}]
    ##############
    parameters_to_optimiza.append({"params": shape_partial_derivate[0].parameters()})
    #for frame_idx in range(num_frames):
    #    parameters_to_optimiza.append({"params":shape_partial_derivate[frame_idx].parameters()})
    ##############
    parameters_to_optimiza.append({'params': model_derivation})
    parameters_to_optimiza.append({'params': shape_decoder.parameters()})
    learning_rate = 0.0001
    optimizer = to.optim.Rprop(parameters_to_optimiza, lr=learning_rate, step_sizes=(1e-10, 50))
    #schduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3000, threshold=0.0001)
    error_reported = np.zeros(shape=(1, num_iterations), dtype=np.float32)
    
    # [Added] Resume logic for Network 2 (MLP mode)
    start_iter = 0
    ckpt_path = os.path.join(result_folder, "ckpt_network2_mlp_latest.pth") # Different name to distinguish GCN version
    
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming Network 2 (MLP) from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        shape_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        shape_partial_derivate[0].load_state_dict(checkpoint['net1_0_state_dict'])
        # MLP mode might not use net1_1 or have different structure, saving what we can
        # shape_partial_derivate[1].load_state_dict(checkpoint['net1_1_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        with torch.no_grad():
            shape_latent_code.data = checkpoint['latent_code_data']
            model_derivation.data = checkpoint['model_derivation_data']
        start_iter = checkpoint['iteration'] + 1
        print(f"Resumed from iteration {start_iter}")

    try:
        batch_size = 500 # [Added] for gradient accumulation
        for i in range(start_iter, num_iterations):
            ## Training
            optimizer.zero_grad()
            
            # [Added] Gradient Accumulation Loop
            cumulative_loss = 0
            for start_f in range(0, num_frames, batch_size):
                end_f = min(start_f + batch_size, num_frames)
                frame_indices = list(range(start_f, end_f))
                
                # [Modified] Call loss with frame_indices to reduce memory footprint
                loss = all_loss_function.loss_all(shape_decoder, shape_latent_code, shape_partial_derivate, model_derivation, i, frame_indices=frame_indices)
                
                # Normalize loss by accumulation steps if loss is averaged, 
                # but for NRSfM generally we backward directly.
                loss.backward()
                cumulative_loss += loss.item()
                #torch.cuda.empty_cache()

            optimizer.step()
            #schduler.step(loss)

            ## Result evaluation
            # [Modified] Detach depth for evaluation to prevent graph accumulation
            # For evaluation, we still might need the full depth
            val_e3d = 0.0 # [Added] Initialize e3D variable
            
            with torch.no_grad():
                depth = shape_decoder.forward(shape_latent_code)
                depth_eval = depth.detach() # Cut gradient flow
            
            normilized_point_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
            #points_3D_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
            for frame_idx in range(depth.shape[0]):
                normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
                normilized_point_result[frame_idx, 2, :] = np.ones(depth.shape[2])
            
            # Use detached depth for calculation
            points_3D_result = normilized_point_result * depth_eval.cpu().numpy().repeat(3, 1)

            # [Modified] Check if Gth is valid to avoid error, and ensure scalar storage
            if Gth is not None and np.any(Gth != 0):
                err_val = shape_error(points_3D_result, Gth, m)
                if isinstance(err_val, torch.Tensor):
                    err_val = err_val.item()
                error_reported[0,i] = err_val

                # ★ [Added] e3D Calculation Logic ★
                try:
                    Gth_e3d = Gth
                    if Gth.ndim == 2 and Gth.shape[0] == num_frames * 3:
                         Gth_e3d = Gth.reshape(num_frames, 3, num_points)
                    
                    if Gth_e3d.shape == points_3D_result.shape:
                        val_e3d = compute_dense_e3d(points_3D_result, Gth_e3d, do_scale=True)
                except Exception as e_calc:
                    val_e3d = 0.0
            else:
                error_reported[0,i] = 0.0

            if i % 3 == 2:  # print every 3 iterations
                # [Modified] Use cumulative_loss.item() to print scalar value
                # [Added] Added val_e3d print
                print('[%5d, %5d] loss: %.3f accuracy: %.6f | e3D: %.6f' %(i + 1, num_iterations, cumulative_loss, error_reported[0,i], val_e3d))
            
            # [Added] Save Checkpoint every 100 iterations
            if (i + 1) % 3 == 0:
                print(f"Saving Checkpoint at iteration {i+1}...")
                torch.save({
                    'iteration': i,
                    'decoder_state_dict': shape_decoder.state_dict(),
                    'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                    # 'net1_1_state_dict': shape_partial_derivate[1].state_dict(), 
                    'latent_code_data': shape_latent_code.data,
                    'model_derivation_data': model_derivation.data,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)
                # Save intermediate depth
                torch.save(depth, os.path.join(result_folder, "depth_latest.pt"))
                
                # [Added] Explicit garbage collection
                del depth, depth_eval, points_3D_result
                gc.collect()
                #torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nNetwork 2 (MLP) training interrupted at iteration {i} due to error: {e}")
        print("Attempting to save emergency checkpoint...")
        try:
            torch.save({
                'iteration': i,
                'decoder_state_dict': shape_decoder.state_dict(),
                'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                'latent_code_data': shape_latent_code.data,
                'model_derivation_data': model_derivation.data,
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Emergency checkpoint saved to {ckpt_path}")
        except:
            print("Failed to save emergency checkpoint.")
        raise e

    print('\n\n\n', 'Compiling complete')
    print("starting optimization")
    ## Storing
    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code, os.path.join(result_folder, "shape_latent_code.pt"))
    torch.save(depth, os.path.join(result_folder, "depth.pt"))
    ## View results
    # [Modified] Check if Gth is valid
    if Gth is not None and np.any(Gth != 0):
        # Re-calculate points_3D_result if needed because we deleted it
        depth_final = shape_decoder.forward(shape_latent_code).detach()
        points_3D_final = normilized_point_result * depth_final.cpu().numpy().repeat(3, 1)
        error_reported[0, i] = shape_error_image(points_3D_final, Gth, m)

    return 1

# [Modified] Added resume parameter
def train_shape_decoder_GCN(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, device, resume=False):
    normilized_point_batched,normilized_point_batched_tensor=get_batched_W(normilized_point, device)
    num_frames=normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]
    num_iterations=5000
    kNN_degree=20
    shape_latent_code = to.zeros((num_frames, 1), requires_grad=True, dtype=torch.float32, device=device)
    network_model = "MLP"
    if network_model== "MLP":
        shape_decoder = ShapeDecoder(num_frames, num_points, Initial_shape, device).to(device)
    elif network_model== "DGNC":
        shape_decoder = ShapeDecoder_DGNC(num_points, num_points=20, mode=0).to(device)
    shape_partial_derivate = model_shape
    all_loss_function = NRSfMLoss(normilized_point_batched, num_points, J, m, device, degree=kNN_degree, normilized_point=normilized_point) # degree is the
    if network_model == "MLP":
        parameters_to_optimiza = [{'params': shape_latent_code}]
    elif network_model == "DGNC":
        parameters_to_optimiza = []
        shape_latent_code = to.zeros((num_frames, 3, num_points), requires_grad=False, dtype=torch.float32, device=device)
        shape_latent_code[:, [0,1], :] = normilized_point_batched_tensor
    #for i in range(2):
    #    shape_partial_derivate[i].requires_grad = True
    #    parameters_to_optimiza.append({"params": shape_partial_derivate[i].parameters()})
    parameters_to_optimiza.append({'params': shape_decoder.parameters()})
    learning_rate = 0.0001
    optimizer = to.optim.Rprop(parameters_to_optimiza, lr=learning_rate, step_sizes=(1e-10, 50))
    #optimizer = torch.optim.Adam(parameters_to_optimiza, lr=learning_rate)
    error_reported = np.zeros(shape=(1, num_iterations), dtype=np.float32)
    
    # [Added] Resume logic for Network 2 (GCN)
    start_iter = 0
    ckpt_path = os.path.join(result_folder, "ckpt_network2_latest.pth")
    
    if resume and os.path.exists(ckpt_path):
        print(f"Resuming Network 2 (GCN) from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 恢复状态
        shape_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        shape_partial_derivate[0].load_state_dict(checkpoint['net1_0_state_dict'])
        shape_partial_derivate[1].load_state_dict(checkpoint['net1_1_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复 latent code (Tensor)
        with torch.no_grad():
            shape_latent_code.data = checkpoint['latent_code_data']
            
        start_iter = checkpoint['iteration'] + 1
        print(f"Resumed from iteration {start_iter}")

    try:
        batch_size =30 # [Added] for gradient accumulation
        for i in range(start_iter, num_iterations):
            shape_partial_derivate[0].train()
            shape_partial_derivate[1].train()
            shape_decoder.train()
            optimizer.zero_grad()
            
            # [Added] Gradient Accumulation Loop
            cumulative_loss = 0
            for start_f in range(0, num_frames, batch_size):
                end_f = min(start_f + batch_size, num_frames)
                frame_indices = list(range(start_f, end_f))
                
                #torch.cuda.empty_cache()
                # [Modified] Call loss with frame_indices
                loss = all_loss_function.loss_all_GNC(shape_decoder, shape_latent_code, shape_partial_derivate, i, network_model, torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device), frame_indices=frame_indices)
                loss.backward()#retain_graph=True
                cumulative_loss += loss.item()

            optimizer.step()
            #schduler.step(loss)
            shape_partial_derivate[0].eval()
            shape_partial_derivate[1].eval()
            shape_decoder.eval()

            ## Result evaluation
            # [Modified] Detach depth for evaluation
            val_e3d = 0.0 # [Added] Initialize e3D variable
            
            with torch.no_grad():
                depth = shape_decoder.forward(shape_latent_code)
            
            if network_model == "MLP":
                depth_eval = depth.detach()
            elif network_model == "DGNC":
                depth_eval = depth.detach() + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
                depth_eval = torch.unsqueeze(depth_eval, 1)
            
            normilized_point_result = np.zeros(shape=(depth_eval.shape[0], 3, depth_eval.shape[2]), dtype=np.float32)
            #points_3D_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
            for frame_idx in range(depth_eval.shape[0]):
                normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
                normilized_point_result[frame_idx, 2, :] = np.ones(depth_eval.shape[2])
            
            points_3D_result = normilized_point_result * depth_eval.cpu().numpy().repeat(3, 1)

            # [Modified] Check if Gth is valid to avoid error, and ensure scalar storage
            if Gth is not None and np.any(Gth != 0):
                err_val = shape_error(points_3D_result, Gth, m)
                if isinstance(err_val, torch.Tensor):
                    err_val = err_val.item()
                error_reported[0,i] = err_val

                # ★ [Added] e3D Calculation Logic ★
                '''try:
                    Gth_e3d = Gth
                    if Gth.ndim == 2 and Gth.shape[0] == num_frames * 3:
                        Gth_e3d = Gth.reshape(num_frames, 3, num_points)
                    elif Gth.ndim == 3:
                        Gth_e3d = Gth
                    if Gth_e3d.shape == points_3D_result.shape:
                        val_e3d = compute_dense_e3d(points_3D_result, Gth_e3d, do_scale=True)
                except Exception as e_calc:
                    val_e3d = 0.0'''
                
                try:
                    Gth_e3d = Gth
                    # 确保 GT 维度正确
                    if Gth.ndim == 2 and Gth.shape[0] == num_frames * 3:
                        Gth_e3d = Gth.reshape(num_frames, 3, num_points)
                    elif Gth.ndim == 3:
                        Gth_e3d = Gth
                    
                    # 检查形状匹配
                    if Gth_e3d.shape == points_3D_result.shape:
                        # !!! 关键修改: 移除了 do_scale=True, 增加了 outlier_threshold
                        # Paper: 100.0, T-shirt: 50.0
                        val_e3d = compute_dense_e3d(points_3D_result, Gth_e3d, outlier_threshold=70.0)
                    else:
                        print(f"e3D Error: Shape mismatch Pred{points_3D_result.shape} vs GT{Gth_e3d.shape}")
                        val_e3d = 0.0
                except Exception as e_calc:
                    # 打印具体报错，不再静默失败
                    print(f"e3D Calculation Failed: {e_calc}")
                    val_e3d = 0.0

            else:
                error_reported[0,i] = 0.0

            #if i % 3 == 2:  # print every 3 iterations
            if i % 5 == 0:
                # [Modified] Use cumulative_loss
                # [Added] Added val_e3d print
                print('[%5d, %5d] loss: %.3f accuracy: %.6f | e3D: %.6f' %(i + 1, num_iterations, cumulative_loss, error_reported[0,i], val_e3d))

           
            if  i % 500 == 0: 
                depth_filename = f"depth_{i}.pt"
                depth_save_path = os.path.join(result_folder, depth_filename)
                torch.save(depth, depth_save_path)
                print(f"Saved depth file for iteration {i} to {depth_save_path}")
            


            # [Added] Save Checkpoint every 100 iterations
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
                # Save intermediate depth
                torch.save(depth, os.path.join(result_folder, "depth_latest.pt"))
                
                # [Added] Explicit garbage collection
                del depth, depth_eval, points_3D_result
                gc.collect()
                #torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nNetwork 2 (GCN) training interrupted at iteration {i} due to error: {e}")
        print("Attempting to save emergency checkpoint...")
        try:
            torch.save({
                'iteration': i,
                'decoder_state_dict': shape_decoder.state_dict(),
                'net1_0_state_dict': shape_partial_derivate[0].state_dict(),
                'net1_1_state_dict': shape_partial_derivate[1].state_dict(),
                'latent_code_data': shape_latent_code.data,
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Emergency checkpoint saved to {ckpt_path}")
        except:
            print("Failed to save emergency checkpoint.")
        raise e

    print('\n\n\n', 'Compiling complete')
    print("starting optimization")
    ## Storing
    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code, os.path.join(result_folder, "shape_latent_code.pt"))
    #多加的几行避免 local variable 'depth' referenced before assignment报错。按理来说就是多存一遍depth_latest.pt
    with torch.no_grad():
    
        depth_final = shape_decoder.forward(shape_latent_code)

        if network_model == "DGNC":
            depth_final = depth_final + torch.tensor(
                Initial_shape,
                requires_grad=False,
                dtype=torch.float32
            ).to(device)
            depth_final = torch.unsqueeze(depth_final, 1)

    torch.save(depth_final, os.path.join(result_folder, "depth_noglobal.pt"))

    ## View results
    # [Modified] Check if Gth is valid
    if Gth is not None and np.any(Gth != 0):
        # Re-calculate points_3D_result if needed because we deleted it
        depth_final = shape_decoder.forward(shape_latent_code).detach()
        if network_model == "DGNC":
             depth_final = depth_final + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
             depth_final = torch.unsqueeze(depth_final, 1)
        
        points_3D_final = normilized_point_result * depth_final.cpu().numpy().repeat(3, 1)
        error_reported[0, i] = shape_error_image(points_3D_final, Gth, m)
        
    final_eval_error = error_reported[0, i]
    print(f"\nFinal Evaluation Accuracy (shape_error_image): {final_eval_error:.6f}")

    return 1