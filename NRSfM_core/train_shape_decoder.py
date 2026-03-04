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

# [Added] Function to compute e3D error based on Paper [14]
def compute_dense_e3d(prediction, ground_truth, do_scale=True):
    """
    Compute e3D error: (1/T) * Sum( ||S_gt - S_recon||_F / ||S_gt||_F )
    Includes Orthogonal Procrustes alignment (with optional scaling).
    """
    F, _, P = prediction.shape
    error_sum = 0.0
    
    for t in range(F):
        P_pred = prediction[t] # (3, P)
        P_gt = ground_truth[t] # (3, P)
        
        # 1. Center data
        mu_pred = P_pred.mean(axis=1, keepdims=True)
        mu_gt = P_gt.mean(axis=1, keepdims=True)
        A = P_pred - mu_pred
        B = P_gt - mu_gt
        
        # 2. Procrustes Alignment (Find R)
        # Min || B - s * R * A ||
        U, S, Vt = np.linalg.svd(B @ A.T)
        R = U @ Vt
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
            
        # 3. Calculate Scale s (if enabled)
        if do_scale:
            A_rotated = R @ A
            nom = np.sum(A_rotated * B)
            denom = np.sum(A_rotated * A_rotated)
            s = nom / (denom + 1e-10)
        else:
            s = 1.0
            
        # 4. Compute Frobenius Norm Error
        # Error = || B - s*R*A ||_F / || B ||_F
        diff_norm = np.linalg.norm(B - s * (R @ A), 'fro')
        gt_norm = np.linalg.norm(B, 'fro')
        
        error_sum += diff_norm / (gt_norm + 1e-10)
        
    return error_sum / F

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
            val_e3d = 0.0 # Initialize e3D value
            if Gth is not None and np.any(Gth != 0):
                err_val = shape_error(points_3D_result, Gth, m)
                if isinstance(err_val, torch.Tensor):
                    err_val = err_val.item()
                error_reported[0,i] = err_val

                # [Added] Calculate e3D error for dense dataset
                try:
                    # Handle dimension mismatch: Gth might be (3F, P) or (F, 3, P)
                    Gth_e3d = Gth
                    if Gth.ndim == 2 and Gth.shape[0] == num_frames * 3:
                         Gth_e3d = Gth.reshape(num_frames, 3, num_points)
                    elif Gth.ndim == 3:
                         Gth_e3d = Gth
                    
                    if Gth_e3d.shape == points_3D_result.shape:
                        val_e3d = compute_dense_e3d(points_3D_result, Gth_e3d, do_scale=True)
                except Exception as e_calc:
                    # In case of dimension mismatch or other errors, suppress and continue
                    val_e3d = 0.0
            else:
                error_reported[0,i] = 0.0

            if i % 10 == 0:  # print every 3 iterations
                # [Modified] Use cumulative_loss.item() to print scalar value
                # [Modified] Added e3D print
                print('[%5d, %5d] loss: %.3f accuracy: %.6f | e3D: %.6f' %(i + 1, num_iterations, cumulative_loss, error_reported[0,i], val_e3d))
            
            # [Added] Save Checkpoint every 100 iterations
            if (i + 1) % 100 == 0:
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

    normilized_point_result_cached = np.zeros(shape=(num_frames, 3, num_points), dtype=np.float32)#预构建
    for frame_idx in range(num_frames):
        normilized_point_result_cached[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
        normilized_point_result_cached[frame_idx, 2, :] = np.ones(num_points)

    try:
        batch_size =20 # [Added] for gradient accumulation
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
                        ## Result evaluation — 每 eval_interval 次迭代才调MATLAB评估，其余只打印loss
            eval_interval = 10
            do_eval = ((i + 1) % eval_interval == 0) or (i == start_iter)
            val_e3d = 0.0

            if do_eval:
                with torch.no_grad():
                    depth_eval_raw = shape_decoder.forward(shape_latent_code)
                
                if network_model == "MLP":
                    depth_eval = depth_eval_raw.detach()
                elif network_model == "DGNC":
                    depth_eval = depth_eval_raw.detach() + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
                    depth_eval = torch.unsqueeze(depth_eval, 1)
                
                points_3D_result = normilized_point_result_cached * depth_eval.cpu().numpy().repeat(3, 1)

                if Gth is not None and np.any(Gth != 0):
                    err_val = shape_error(points_3D_result, Gth, m)
                    if isinstance(err_val, torch.Tensor):
                        err_val = err_val.item()
                    error_reported[0, i] = err_val

                    try:
                        Gth_e3d = Gth
                        if Gth.ndim == 2 and Gth.shape[0] == num_frames * 3:
                            Gth_e3d = Gth.reshape(num_frames, 3, num_points)
                        elif Gth.ndim == 3:
                            Gth_e3d = Gth
                        if Gth_e3d.shape == points_3D_result.shape:
                            val_e3d = compute_dense_e3d(points_3D_result, Gth_e3d, do_scale=True)
                    except Exception as e_calc:
                        val_e3d = 0.0
                else:
                    error_reported[0, i] = 0.0

            if (i + 1) % 10 == 0:
                if do_eval:
                    print('[%5d, %5d] loss: %.3f accuracy: %.6f | e3D: %.6f' % (i + 1, num_iterations, cumulative_loss, error_reported[0, i], val_e3d))
                else:
                    print('[%5d, %5d] loss: %.3f' % (i + 1, num_iterations, cumulative_loss))

            
            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    depth_for_save = shape_decoder.forward(shape_latent_code)
                depth_filename = f"depth_{i}.pt"
                depth_save_path = os.path.join(result_folder, depth_filename)
                torch.save(depth_for_save, depth_save_path)
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
                torch.save(depth_for_save, os.path.join(result_folder, "depth_latest.pt"))
                
                gc.collect()

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

    torch.save(depth_final, os.path.join(result_folder, "depth.pt"))

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