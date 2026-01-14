import numpy as np
import torch
# Make MATLAB optional
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None
from NRSfM_core.GNN_model import Non_LinearGNN
from NRSfM_core.model_develop_init import Fully_connection
from NRSfM_core.KNN_graph import Graph_distance
from NRSfM_core.DGCN_model import DGCNNControlPoints
from NRSfM_core.loss_DGCN import (
    uniform_knot_bspline,
    spline_reconstruction_loss_one_sided,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from NRSfM_core.fitting_utils import sample_points_from_control_points_
from configobj import ConfigObj
import matplotlib.pyplot as plt
from NRSfM_core.loss_DGCN import (
    control_points_permute_reg_loss,
    laplacian_loss,
)
from NRSfM_core.dataset import (DataSetControlPointsPoisson, generator_iter)
from torch.utils.data import DataLoader
from torch.autograd import Variable


def Initial_supervised_learning_DGCN(Initial_shape, normilized_point_batched, m, kNN_degree, num_iterations):
    ##############################parameters##############################################################
    omega = 0.1
    num_frames = normilized_point_batched.shape[0] // 2
    num_point_per_frame = normilized_point_batched.shape[1]
    num_data = 10
    ##############################parameters##############################################################
    random_depth_data = torch.zeros(num_data*num_frames, num_point_per_frame)
    #shape_partial_derivate = Non_LinearGNN(num_point_per_frame, feat_dim=5, stat_dim=3, iteration=1, degree=kNN_degree)
    #shape_partial_derivate = Fully_connection(num_point_per_frame*3, num_point_per_frame*2)
    normilized_point_result = np.zeros(shape=(num_frames, 3, num_point_per_frame), dtype=np.float32)
    for frame_idx in range(num_frames):
        normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[[frame_idx*2,frame_idx*2+1], :]
        normilized_point_result[frame_idx, 2, :] = np.ones(num_point_per_frame)
    normilized_point = torch.tensor(normilized_point_result)
    '''for data_id in range(num_data):
        for image_id in range(num_frames):
            random_depth_data[data_id*num_frames+image_id, :] = torch.tensor(Low_Bound[image_id, :]+(Up_Bound[image_id, :] - Low_Bound[image_id, :])*np.random.rand(Low_Bound.shape[1]))
            uv = matlab.double(normilized_point[image_id, [0, 1], :].tolist())
            #points_3D = torch.unsqueeze(normilized_point[image_id, [0,1,2],:],0) * random_depth_data[data_id * num_frames + image_id, :].repeat(1, 3, 1)
            points_3D = normilized_point[image_id, [0,1,2],:] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
            fit_result = m.fit_python(uv, matlab.double(points_3D.tolist()), uv, nargout=6)
            dqu = np.array(fit_result[1])
            dqv = np.array(fit_result[2])
            y1_ground[data_id*num_frames+image_id, :] = np.array(dqu)[2, :] / points_3D[2, :]  #Ground truth
            y2_ground[data_id*num_frames+image_id, :] = np.array(dqv)[2, :] / points_3D[2, :]'''
    ###################################################################################################################################################

    grid_size = 20
    dense_point_size=40
    nu, nv = uniform_knot_bspline(grid_size, grid_size, 3, 3, dense_point_size)
    nu = torch.from_numpy(nu.astype(np.float32)).cuda()
    nv = torch.from_numpy(nv.astype(np.float32)).cuda()
    Low_Bound = Initial_shape - np.ones(Initial_shape.shape) * omega
    Up_Bound = Initial_shape + np.ones(Initial_shape.shape) * omega
    gt_points_tensor = torch.zeros(num_data*num_frames, 2000, 3).cuda()
    gt_points_tensor_test = torch.zeros(num_frames, 2000, 3).cuda()
    quv_control_points = torch.zeros(num_data * num_frames, grid_size, grid_size, 3).cuda()


    epochs = 100
    batch_size = 8
    loss_weight = 0.9

    dataset = DataSetControlPointsPoisson(
        "E:\Github\data\data\spline\open_splines.h5",
        batch_size,
        splits={"train": 3200, "val": 3000, "test": 3000},
        size_v=20,
        size_u=20)

    get_train_data = dataset.load_train_data(
        if_regular_points=True, align_canonical=True, anisotropic=True, if_augment=True
    )
    loader = generator_iter(get_train_data, int(1e10))
    get_train_data = iter(
        DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
            pin_memory=False,
        )
    )

    for data_id in range(num_data):
        for image_id in range(num_frames):
            #points_, parameters, control_points, scales, _ = next(get_train_data)[0]

            ''' fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            control_points1 = control_points#.cpu().detach().numpy()  # quv_control_points1
            gt_points = points_#.cpu().detach().numpy()
            for i in range(1):
                for j in range(20):
                    ax.scatter(control_points1[data_id * num_frames + image_id, i, j, 0],
                               control_points1[data_id * num_frames + image_id, i, j, 1],
                               control_points1[data_id * num_frames + image_id, i, j, 2])  # control_points
            # ax.scatter(points_3D[0,:], points_3D[1,:], points_3D[2,:])
            ax.scatter(gt_points[0, :, 0], gt_points[0, :, 1], gt_points[0, :, 2])  # points
            plt.show()'''

            random_depth_data[data_id*num_frames+image_id, :] = torch.tensor(Low_Bound[image_id, :]+(Up_Bound[image_id, :] - Low_Bound[image_id, :])*np.random.rand(Low_Bound.shape[1]))
            umax = max(normilized_point[image_id, 0, :])
            umin = min(normilized_point[image_id, 0, :])
            vmax = max(normilized_point[image_id, 1, :])
            vmin = min(normilized_point[image_id, 1, :])
            control_points = torch.zeros(2, grid_size**2)
            for u_i in range(grid_size):
                for v_i in range(grid_size):
                    control_points[0, u_i * grid_size + v_i] = umin + (umax - umin)/(grid_size-1) * u_i
                    control_points[1, u_i * grid_size + v_i] = vmin + (vmax - vmin)/(grid_size-1) * v_i
            uv = matlab.double(normilized_point[image_id, [0, 1], :].tolist())
            points_3D = normilized_point[image_id, [0,1,2],:] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
            control_points_matlab =  matlab.double(control_points.tolist())
            fit_result = m.fit_python(uv, matlab.double(points_3D.tolist()), control_points_matlab, nargout=6)

            quv = torch.transpose(torch.tensor(np.array(fit_result[0])),1,0).unsqueeze(0)
            quv=quv.to(torch.float32).cuda()
            for i in range(grid_size):
                for j in range(grid_size):
                    quv_control_points[data_id * num_frames + image_id, i, j, :] = quv[:, i*grid_size+ j, :]
            #quv_control_points[data_id*num_frames+image_id, :, :, :] = quv.reshape((1, grid_size, grid_size, 3))#quv[:, i*grid_size+ j, :]
            #control_points = Variable(torch.from_numpy(control_points.astype(np.float32))).cuda()

            gt_points_uv = torch.zeros(2, 2000)
            for i in range(2000):
                gt_points_uv[0, i] = umin + (umax - umin) * torch.tensor(np.random.uniform(0, 1))
                gt_points_uv[1, i] = vmin + (vmax - vmin) * torch.tensor(np.random.uniform(0, 1))
            gt_points_uv_matlab = matlab.double(gt_points_uv.tolist())
            fit_result_uv = m.fit_python(uv, matlab.double(points_3D.tolist()), gt_points_uv_matlab, nargout=6)
            gt_points = torch.transpose(torch.tensor(np.array(fit_result_uv[0])),1,0).unsqueeze(0).to(torch.float32).cuda()
            gt_points_sample = sample_points_from_control_points_(nu, nv, quv_control_points[data_id*num_frames+image_id, :, :, :].unsqueeze(0), 1, input_size_u=grid_size, input_size_v=grid_size).data.cpu().numpy()
            #gt_points = sample_points_from_control_points_(nu, nv, control_points, 1, input_size_u=grid_size, input_size_v=grid_size).data.cpu().numpy()
            gt_points_tensor[data_id * num_frames + image_id, :, :] = gt_points
            #gt_points_tensor[data_id*num_frames+image_id,:,:] = torch.from_numpy(gt_points.astype(np.float32)).cuda()
            '''gt_points_tensor[data_id * num_frames + image_id, :, :] = torch.from_numpy(gt_points.astype(np.float32)).cuda()
            if data_id==0:
                gt_points_tensor_test[image_id, :, :] = torch.from_numpy(gt_points.astype(np.float32)).cuda()'''
            '''fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #for i in range(dense_point_size):
            #    for j in range(dense_point_size):
            #control_points1 = control_points.cpu().detach().numpy()
            control_points1 = quv_control_points.cpu().detach().numpy()#quv_control_points1
            #points_3D = points_3D.cpu().detach().numpy()
            gt_points = gt_points.cpu().detach().numpy()
            for i in range(1):
                for j in range(20):
                    ax.scatter(control_points1[data_id * num_frames + image_id,i,j,0], control_points1[data_id * num_frames + image_id,i,j,1], control_points1[data_id * num_frames + image_id,i,j,2]) #control_points
            #ax.scatter(points_3D[0,:], points_3D[1,:], points_3D[2,:])
            ax.scatter(gt_points[0, :, 0], gt_points[0, :, 1], gt_points[0, :, 2]) #points
            ax.scatter(gt_points_sample[0, :, 0], gt_points_sample[0, :, 1], gt_points_sample[0, :, 2]) #points
            plt.show()'''

        control_decoder = DGCNNControlPoints(20, num_points=10, mode=0)
        control_decoder = torch.nn.DataParallel(control_decoder)
        control_decoder.cuda()
        control_decoder.load_state_dict(
            torch.load("E:/Github/data/logs/pretrained_models/" + "open_spline.pth")
            #torch.load("E:/Github/data/logs/pretrained_models/" + "closed_spline.pth")
        )
        normilized_point_test = np.zeros(shape=(num_frames, 3, num_point_per_frame), dtype=np.float32)
        point_used = torch.zeros(num_frames, 3, num_point_per_frame)
        for frame_idx in range(num_frames):
            normilized_point_test[frame_idx, [0,1], :] = normilized_point_batched[[frame_idx*2,frame_idx*2+1], :]
            normilized_point_test[frame_idx, 2, :] = np.ones(num_point_per_frame)
            point_used[frame_idx, :, :] = torch.tensor(normilized_point_test[frame_idx, :, :])* torch.tensor(Initial_shape[frame_idx, :]).repeat(3,1)
            point_used[frame_idx, 0, :] = 0.1*point_used[frame_idx, 0, :]
            point_used[frame_idx, 1, :] = 0.1*point_used[frame_idx, 1, :]
            point_used[frame_idx, 2, :] = 0.05*point_used[frame_idx, 2, :]
        output = control_decoder(point_used)
        gt_points_sample = sample_points_from_control_points_(nu, nv, output, 1, input_size_u=grid_size, input_size_v=grid_size).data.cpu().numpy()
        output = output.cpu().detach().numpy()
        for frame_idx in range(num_frames):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_points_sample[frame_idx, :, 0], gt_points_sample[frame_idx, :, 1], gt_points_sample[frame_idx, :, 2])
            ax.scatter(output[frame_idx, :, 0], output[frame_idx, :, 1], output[frame_idx, :, 2])
            ax.scatter(point_used[frame_idx, 0, :], point_used[frame_idx, 1, :], point_used[frame_idx, 2, :])
            plt.show()
            aaa=1

    ##############################parameters##############################################################
    shape_partial_derivate = DGCNNControlPoints(grid_size, num_points=20, mode=0)
    if torch.cuda.device_count() > 1:
        shape_partial_derivate = torch.nn.DataParallel(shape_partial_derivate)
    shape_partial_derivate.cuda()
    optimizer = torch.optim.Rprop(shape_partial_derivate.parameters(), lr=0.0001, step_sizes=(1e-10, 50))
    # optimizer = torch.optim.Adam(shape_partial_derivate.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=3e-5)

    for e in range(epochs):
        shape_partial_derivate.train()
        loss=torch.zeros(1,1).cuda()
        for data_id in range(num_data*num_frames // batch_size):
            '''points_, parameters, control_points, scales, _ = next(get_train_data)[0]#~~'''

            ## Training
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            rand_num_points = Initial_shape.shape[1] +  np.random.choice(np.arange(-Initial_shape.shape[1] * 0.2, Initial_shape.shape[1] * 0.2), 1)[0]
            #rand_num_points =

            #points_pick_out = gt_points_tensor[data_id*batch_size:(data_id+1)*batch_size, 0:rand_num_points.astype(int), :]
            points_pick_out_all = gt_points_tensor[data_id * batch_size:(data_id + 1) * batch_size, :, :] #points
            '''points_pick_out_all = torch.from_numpy(points_.astype(np.float32)).cuda()#~~'''
            points_pick_out = points_pick_out_all[ :, 0:rand_num_points.astype(int), :]#~~
            '''fig = plt.figure()
            fig1 = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ay = fig1.add_subplot(111, projection='3d')
            points_pick_out_all1 = points_pick_out_all1.cpu().detach().numpy()
            points_pick_out_all = points_pick_out_all.cpu().detach().numpy()
            ax.scatter(points_pick_out_all1[0,:,0], points_pick_out_all1[0,:,1], points_pick_out_all1[0,:,2]) #control_points
            ay.scatter(points_pick_out_all[0,:,0], points_pick_out_all[0,:,1], points_pick_out_all[0,:,2]) #control_points
            plt.show()'''

            output = shape_partial_derivate(torch.transpose(points_pick_out, 2, 1))
            # Sample random number of points to make network robust to density.
            # Chamfer Distance loss, between predicted and GT surfaces
            #control_points = torch.from_numpy(control_points.astype(np.float32)).cuda()
            control_points =quv_control_points[data_id * batch_size:(data_id + 1) * batch_size, :, :,:]

            cd, reconstructed_points = spline_reconstruction_loss_one_sided(nu, nv, output, torch.transpose(points_pick_out_all, 2, 1), batch_size, grid_size)

            # Permutation Regression Loss
            # permute_cp has the best permutation of gt control points grid
            l_reg, permute_cp = control_points_permute_reg_loss(output, control_points, grid_size)#~~
            '''fig2 = plt.figure()
            fig3 = plt.figure()
            ax = fig2.add_subplot(111, projection='3d')
            ay = fig3.add_subplot(111, projection='3d')
            control_points = control_points.cpu().detach().numpy()
            control_points1 = control_points1.cpu().detach().numpy()
            ax.scatter(control_points[0,:,:,0], control_points[0,:,:,1], control_points[0,:,:,2]) #control_points
            ay.scatter(control_points1[0,:,:,0], control_points1[0,:,:,1], control_points1[0,:,:,2]) #control_points
            plt.show()'''

            laplac_loss = laplacian_loss(output.reshape((batch_size, grid_size, grid_size, 3)), permute_cp, dist_type="l2",)

            loss =loss+ l_reg * loss_weight + (cd + laplac_loss) * (1 - loss_weight)


            loss.backward()
            optimizer.step()

            print(
                "\rEpoch: {} iter: {}, loss: {}".format(
                    e, data_id, loss.item()
                ),
                end="",
            )
    #Low_Bound_all = min(Initial_shape)-(max(Initial_shape)-min(Initial_shape))*0.1
    #    Up_Bound_all = max(Initial_shape)+(max(Initial_shape)-min(Initial_shape))*0.1
        shape_partial_derivate.eval()

    '''for val_b_id in range(num_frames // batch_size):
        output = shape_partial_derivate(torch.transpose(gt_points_tensor_test[val_b_id * batch_size:(val_b_id + 1) * batch_size - 1, 0:rand_num_points.astype(int), :], 2, 1))
        # Sample random number of points to make network robust to density.
        cd, reconstructed_points = spline_reconstruction_loss_one_sided(nu, nv, output, gt_points_tensor, batch_size,  dense_point_size)

        if i % 3 == 2:  # print every 3 iterations
            print('[%5d, %5d] loss: %.3f %.3f' % (i + 1, num_iterations, output, accuracy))'''

    return 1 #shape_weights