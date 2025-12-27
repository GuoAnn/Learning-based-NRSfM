import numpy as np
import torch
import matlab.engine
from NRSfM_core.GNN_model import Non_LinearGNN
from NRSfM_core.model_develop_init import Fully_connection
from NRSfM_core.KNN_graph import Graph_distance

def Initial_supervised_learning(Initial_shape, normilized_point_batched, m, kNN_degree, num_iterations):
##############################parameters##############################################################
    omega = 0.3
    num_frames = 1#normilized_point_batched.shape[0] // 2
    num_point_per_frame = normilized_point_batched.shape[1]
    num_data = 1
##############################parameters##############################################################
    random_depth_data = torch.zeros(num_data*num_frames, num_point_per_frame)
    shape_partial_derivate = Non_LinearGNN(num_point_per_frame, feat_dim=5, stat_dim=3, iteration=1, degree=kNN_degree)
    #shape_partial_derivate = Fully_connection(num_point_per_frame*3, num_point_per_frame*2)
    ID = Graph_distance(normilized_point_batched[[0, 1], :], kNN_degree)
    node_neis = ID.flatten()
    node_inds = np.repeat(np.arange(num_point_per_frame), kNN_degree)
    X_Node = torch.tensor(node_inds)
    X_Neis = torch.tensor(node_neis)
    normilized_point_result = np.zeros(shape=(num_frames, 3, num_point_per_frame), dtype=np.float32)
    Low_Bound = Initial_shape-np.ones(Initial_shape.shape)*omega
    Up_Bound = Initial_shape+np.ones(Initial_shape.shape)*omega
    y1_ground = torch.zeros(num_data*num_frames, num_point_per_frame)
    y2_ground = torch.zeros(num_data*num_frames, num_point_per_frame)
    for frame_idx in range(num_frames):
        normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[[frame_idx*2,frame_idx*2+1], :]
        normilized_point_result[frame_idx, 2, :] = np.ones(num_point_per_frame)
    normilized_point = torch.tensor(normilized_point_result)
    for data_id in range(num_data):
        for image_id in range(num_frames):
            random_depth_data[data_id*num_frames+image_id, :] = torch.tensor(Low_Bound[image_id, :]+(Up_Bound[image_id, :] - Low_Bound[image_id, :])*np.random.rand(Low_Bound.shape[1]))
            uv = matlab.double(normilized_point[image_id, [0, 1], :].tolist())
            #points_3D = torch.unsqueeze(normilized_point[image_id, [0,1,2],:],0) * random_depth_data[data_id * num_frames + image_id, :].repeat(1, 3, 1)
            points_3D = normilized_point[image_id, [0,1,2],:] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
            fit_result = m.fit_python(uv, matlab.double(points_3D.tolist()), uv, nargout=6)
            dqu = np.array(fit_result[1])
            dqv = np.array(fit_result[2])
            y1_ground[data_id*num_frames+image_id, :] = np.array(dqu)[2, :] / points_3D[2, :]  #Ground truth
            y2_ground[data_id*num_frames+image_id, :] = np.array(dqv)[2, :] / points_3D[2, :]
###################################################################################################################################################
    parameters_to_optimiza = [{"params":shape_partial_derivate.parameters()}]
    optimizer = torch.optim.Rprop(parameters_to_optimiza, lr=0.0001, step_sizes=(1e-10, 50))
    loss = torch.nn.MSELoss()#torch.nn.L1Loss()
    for i in range(num_iterations):
        ## Training
        optimizer.zero_grad()
        #y1_predict = torch.zeros(num_data * num_frames, num_point_per_frame)
        #y2_predict = torch.zeros(num_data * num_frames, num_point_per_frame)
        output=torch.zeros(1)
        for data_id in range(num_data):
            for image_id in range(num_frames):
                points_3D = normilized_point[image_id, [0, 1, 2], :] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
                #y_result = shape_partial_derivate.forward(X_Node, X_Neis, points_3D)
                y_result = shape_partial_derivate.forward(X_Node, X_Neis, torch.cat((points_3D, normilized_point[image_id, [0, 1, ], :]), 0))
                #y_result = shape_partial_derivate.forward(torch.flatten(points_3D))
                #y1_predict[data_id*num_frames+image_id, :] = y_result[:num_point_per_frame]
                #y2_predict[data_id*num_frames+image_id, :] = y_result[num_point_per_frame:]
                loss1 = loss(y_result[:num_point_per_frame], y1_ground[data_id*num_frames+image_id, :])
                loss2 = loss(y_result[num_point_per_frame:], y2_ground[data_id*num_frames+image_id, :])
                output = loss1  + output + loss2

        #loss2 = loss(y2_predict, y2_ground)

        output.backward(retain_graph=True)  # retain_graph=True
        optimizer.step()

        points_3D_ref = normilized_point[0, [0, 1, 2], :] * random_depth_data[0, :].repeat(3, 1)
        y_result_ref = shape_partial_derivate.forward(X_Node, X_Neis, torch.cat((points_3D_ref, normilized_point[0, [0, 1], :]), 0))
        #y_result_ref = shape_partial_derivate.forward(X_Node, X_Neis, points_3D_ref)
        #y_result_ref = shape_partial_derivate.forward( torch.flatten(points_3D_ref))
        y1_predict_ref = y_result_ref[:num_point_per_frame]
        y2_predict_ref = y_result_ref[num_point_per_frame:]
        accuracy = torch.sum(((y1_predict_ref - y1_ground[0,:])+(y2_predict_ref - y2_ground[0,:]))**2)

        if i % 3 == 2:  # print every 3 iterations
            print('[%5d, %5d] loss: %.3f %.3f' % (i + 1, num_iterations, output, accuracy))
    return 1 #shape_weights