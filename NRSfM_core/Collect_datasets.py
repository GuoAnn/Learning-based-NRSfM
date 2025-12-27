import numpy as np
import torch
import matlab.engine
from NRSfM_core.GNN_model import Non_LinearGNN
from NRSfM_core.model_develop_init import Fully_connection
from NRSfM_core.new_DGCN_model import DGCNNControlPoints
from NRSfM_core.KNN_graph import Graph_distance

def Collect_data(Initial_shape, normilized_point_batched, m, device, num_data):
##############################parameters##############################################################
    omega = 0.2
    num_frames = normilized_point_batched.shape[0] // 2
    num_point_per_frame = normilized_point_batched.shape[1]
##############################parameters##############################################################
    random_depth_data = torch.zeros(num_data*num_frames, num_point_per_frame).to(device)
    shape_partial_derivate=[]
##############################parameters##############################################################
    normilized_point_result = np.zeros(shape=(num_frames, 3, num_point_per_frame), dtype=np.float32)
    Low_Bound = Initial_shape-np.ones(Initial_shape.shape)*omega
    Up_Bound = Initial_shape+np.ones(Initial_shape.shape)*omega
    y1_ground = torch.zeros(num_data*num_frames, num_point_per_frame).to(device)
    y2_ground = torch.zeros(num_data*num_frames, num_point_per_frame).to(device)
    for frame_idx in range(num_frames):
        normilized_point_result[frame_idx, [0,1], :] = normilized_point_batched[[frame_idx*2,frame_idx*2+1], :]
        normilized_point_result[frame_idx, 2, :] = np.ones(num_point_per_frame)
    normilized_point = torch.tensor(normilized_point_result).to(device)
    points_3D_all = torch.zeros(num_data*num_frames, 3, num_point_per_frame).to(device)
    for data_id in range(num_data):
        for image_id in range(num_frames):
            random_depth_data[data_id*num_frames+image_id, :] = torch.tensor(Low_Bound[image_id, :]+(Up_Bound[image_id, :] - Low_Bound[image_id, :])*np.random.rand(Low_Bound.shape[1])).to(device)
            uv = matlab.double(normilized_point[image_id, [0, 1], :].tolist())
            #points_3D = torch.unsqueeze(normilized_point[image_id, [0,1,2],:],0) * random_depth_data[data_id * num_frames + image_id, :].repeat(1, 3, 1)
            points_3D = normilized_point[image_id, [0,1,2],:] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
            fit_result = m.fit_python(uv, matlab.double(points_3D.tolist()), uv, nargout=6)
            dqu = torch.tensor(np.array(fit_result[1])).to(device)
            dqv = torch.tensor(np.array(fit_result[2])).to(device)
            y1_ground[data_id*num_frames+image_id, :] = dqu[2, :] / points_3D[2, :]  #Ground truth
            y2_ground[data_id*num_frames+image_id, :] = dqv[2, :] / points_3D[2, :]
            points_3D_all[data_id*num_frames+image_id, [0,1,2],:] =  points_3D
###################################################################################################################################################

    return  points_3D_all, y1_ground, y2_ground#, latent_space #shape_weights

def Initial_learning_from_all_datasets():

    return