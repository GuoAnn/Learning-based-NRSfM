import numpy as np
import torch
import os

from Dataset.load_dataset import get_batched_W
import torch as to
from NRSfM_core.loss_function import NRSfMLoss
#from NRSfM_core.model_develop import learning_model
from NRSfM_core.shape_decoder import ShapeDecoder, ShapeDecoder_DGNC
from Result_evaluation.Shape_error import shape_error, shape_error_image,shape_error_save
from NRSfM_core.GNN_model import Non_LinearGNN

def train_shape_decoder(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, model_derivation, device):
    normilized_point_batched,normilized_point_batched_tensor=get_batched_W(normilized_point, device)
    num_frames=normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]
    num_iterations=50 #100000
    kNN_degree=8
    ## Tensorfolow
    #shape_latent_code = tf.random.normal([num_frames,1], 0, 1, tf.float32, seed=1)
    ## Pytorch
    #shape_latent_code = to.randn((num_frames,1),requires_grad=True, dtype=torch.float32)
    shape_latent_code = to.zeros((num_frames, 1), requires_grad=True, dtype=torch.float32, device=device)
    #May change into num_points parameter: num_points=normilized_point_batched.shape[1]
    shape_decoder = ShapeDecoder(num_frames, num_points, Initial_shape, device).to(device)
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
    for i in range(num_iterations):
        ## Training
        optimizer.zero_grad()
        loss = all_loss_function.loss_all(shape_decoder, shape_latent_code, shape_partial_derivate, model_derivation, i)
        loss.backward()#retain_graph=True
        optimizer.step()
        #schduler.step(loss)

        ## Result evaluation
        depth = shape_decoder.forward(shape_latent_code)
        normilized_point_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
        #points_3D_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
        for frame_idx in range(depth.shape[0]):
            normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
            normilized_point_result[frame_idx, 2, :] = np.ones(depth.shape[2])
        points_3D_result = normilized_point_result * depth.cpu().detach().numpy().repeat(3, 1)

        error_reported[0,i]=shape_error(points_3D_result, Gth, m)

        if i % 3 == 2:  # print every 3 iterations
            print('[%5d, %5d] loss: %.3f accuracy: %.6f' %(i + 1, num_iterations, loss, error_reported[0,i]))

    print('\n\n\n', 'Compiling complete')
    print("starting optimization")
    ## Storing
    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code, os.path.join(result_folder, "shape_latent_code.pt"))
    torch.save(depth, os.path.join(result_folder, "depth.pt"))
    ## View results
    error_reported[0, i] = shape_error_image(points_3D_result, Gth, m)

    return 1

'''
def train_shape_decoder_GCN(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, device):
    normilized_point_batched,normilized_point_batched_tensor=get_batched_W(normilized_point, device)
    num_frames=normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]
    num_iterations=50 #5000
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
    error_reported = np.zeros(shape=(1, num_iterations), dtype=np.float32)
    for i in range(num_iterations):
        shape_partial_derivate[0].train()
        shape_partial_derivate[1].train()
        shape_decoder.train()
        optimizer.zero_grad()
        #torch.cuda.empty_cache() #可能会加重碎片化
        loss = all_loss_function.loss_all_GNC(shape_decoder, shape_latent_code, shape_partial_derivate, i, network_model, torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device))
        loss.backward()#retain_graph=True
        optimizer.step()
        #schduler.step(loss)
        shape_partial_derivate[0].eval()
        shape_partial_derivate[1].eval()
        shape_decoder.eval()

        ## Result evaluation
        depth = shape_decoder.forward(shape_latent_code)
        if network_model == "MLP":
            depth = depth
        elif network_model == "DGNC":
            depth = depth + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
            depth = torch.unsqueeze(depth, 1)
        normilized_point_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
        #points_3D_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
        for frame_idx in range(depth.shape[0]):
            normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[frame_idx, :, :]
            normilized_point_result[frame_idx, 2, :] = np.ones(depth.shape[2])
        points_3D_result = normilized_point_result * depth.cpu().detach().numpy().repeat(3, 1)

        error_reported[0,i]=shape_error(points_3D_result, Gth, m)

        if i % 3 == 2:  # print every 3 iterations
            print('[%5d, %5d] loss: %.3f accuracy: %.6f' %(i + 1, num_iterations, loss, error_reported[0,i]))

    print('\n\n\n', 'Compiling complete')
    print("starting optimization")
    ## Storing
    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code, os.path.join(result_folder, "shape_latent_code.pt"))
    torch.save(depth, os.path.join(result_folder, "depth.pt"))
    ## View results
    error_reported[0, i] = shape_error_image(points_3D_result, Gth, m)

    # 加：逐帧误差并保存
    acc_f, est = shape_error_save(points_3D_result, Gth, m)
    np.save(os.path.join(result_folder, "per_frame_error.npy"), acc_f.cpu().numpy())
    '''

def train_shape_decoder_GCN(result_folder, normilized_point, args, J, m, Initial_shape, Gth, model_shape, device):
    normilized_point_batched, normilized_point_batched_tensor = get_batched_W(normilized_point, device)
    num_frames = normilized_point_batched.shape[0]
    num_points = normilized_point_batched.shape[2]

    # 关键：把 kNN_degree 改成可配置，默认建议 8
    kNN_degree = getattr(args, "knn_k", 8)

    # 关键：把迭代次数也可配置，先跑通再加
    num_iterations = getattr(args, "gcn_iters", 10)

    use_amp = bool(getattr(args, "use_amp", False))
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))

    shape_latent_code = to.zeros((num_frames, 1), requires_grad=True, dtype=torch.float32, device=device)
    network_model = "MLP"

    if network_model == "MLP":
        shape_decoder = ShapeDecoder(num_frames, num_points, Initial_shape, device).to(device)
    elif network_model == "DGNC":
        shape_decoder = ShapeDecoder_DGNC(num_points, num_points=20, mode=0).to(device)

    shape_partial_derivate = model_shape
    all_loss_function = NRSfMLoss(
        normilized_point_batched,
        num_points,
        J,
        m,
        device,
        degree=kNN_degree,
        normilized_point=normilized_point
    )

    parameters_to_optimiza = [{'params': shape_latent_code}]
    parameters_to_optimiza.append({'params': shape_decoder.parameters()})

    optimizer = to.optim.Rprop(parameters_to_optimiza, lr=0.0001, step_sizes=(1e-10, 50))
    error_reported = np.zeros(shape=(1, num_iterations), dtype=np.float32)

    for i in range(num_iterations):
        shape_partial_derivate[0].train()
        shape_partial_derivate[1].train()
        shape_decoder.train()
        optimizer.zero_grad(set_to_none=True)

        # AMP：只包裹 PyTorch 计算部分
        with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
            loss = all_loss_function.loss_all_GNC(
                shape_decoder,
                shape_latent_code,
                shape_partial_derivate,
                i,
                network_model,
                torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
            )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        shape_partial_derivate[0].eval()
        shape_partial_derivate[1].eval()
        shape_decoder.eval()

        # Result evaluation（不需要梯度）
        with torch.no_grad():
            depth = shape_decoder.forward(shape_latent_code)
            if network_model == "DGNC":
                depth = depth + torch.tensor(Initial_shape, requires_grad=False, dtype=torch.float32).to(device)
                depth = torch.unsqueeze(depth, 1)

            normilized_point_result = np.zeros(shape=(depth.shape[0], 3, depth.shape[2]), dtype=np.float32)
            for frame_idx in range(depth.shape[0]):
                normilized_point_result[frame_idx, [0, 1], :] = normilized_point_batched[frame_idx, :, :]
                normilized_point_result[frame_idx, 2, :] = np.ones(depth.shape[2])

            points_3D_result = normilized_point_result * depth.detach().cpu().numpy().repeat(3, 1)
            error_reported[0, i] = shape_error(points_3D_result, Gth, m)

        if i % 3 == 2:
            print('[%5d, %5d] loss: %.3f accuracy: %.6f' % (i + 1, num_iterations, float(loss.detach().cpu()), error_reported[0, i]))

    torch.save(shape_decoder.state_dict(), os.path.join(result_folder, "Model_parameters.pt"))
    torch.save(shape_latent_code.detach().cpu(), os.path.join(result_folder, "shape_latent_code.pt"))
    torch.save(depth.detach().cpu(), os.path.join(result_folder, "depth.pt"))

    error_reported[0, i] = shape_error_image(points_3D_result, Gth, m)
    acc_f, est = shape_error_save(points_3D_result, Gth, m)
    np.save(os.path.join(result_folder, "per_frame_error.npy"), acc_f.cpu().numpy())

    return 1