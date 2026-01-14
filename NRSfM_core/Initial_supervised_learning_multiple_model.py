import numpy as np
import torch
import matlab.engine
from torch.cuda.amp import autocast, GradScaler  # AMP 组件
from NRSfM_core.GNN_model import Non_LinearGNN
from NRSfM_core.model_develop_init import Fully_connection
from NRSfM_core.new_DGCN_model import DGCNNControlPoints
from NRSfM_core.KNN_graph import Graph_distance

def Initial_supervised_learning(Initial_shape, normilized_point_batched, m, device, kNN_degree, num_iterations, num_data):
##############################parameters##############################################################
    omega = 0.1
    num_frames = normilized_point_batched.shape[0] // 2
    num_point_per_frame = normilized_point_batched.shape[1]
##############################parameters##############################################################
    random_depth_data = torch.zeros(num_data*num_frames, num_point_per_frame).to(device)
    shape_partial_derivate=[]
    #shape_partial_derivate.append(Non_LinearGNN(num_point_per_frame, device, feat_dim=5, stat_dim=3, iteration=1, degree=kNN_degree).to(device))
    #shape_partial_derivate.append(Non_LinearGNN(num_point_per_frame, device, feat_dim=5, stat_dim=3, iteration=1, degree=kNN_degree).to(device))
    num_control_points = num_point_per_frame
    shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=12, mode=0).to(device)) #20
    shape_partial_derivate.append(DGCNNControlPoints(num_control_points, num_points=12, mode=0).to(device)) #20

############
    #for frame_idx in range(num_frames): #按帧循环追加多个模型的注释），为每个 frame 都实例化一套网络
    #    shape_partial_derivate.append(Non_LinearGNN(num_point_per_frame, device, feat_dim=5, stat_dim=3, iteration=1, degree=kNN_degree).to(device))
        #shape_partial_derivate.append(Fully_connection(num_point_per_frame*5, num_point_per_frame*2).to(device))
    ############
    ID = Graph_distance(normilized_point_batched[[0, 1], :], kNN_degree)
    node_neis = ID.flatten()
    node_inds = np.repeat(np.arange(num_point_per_frame), kNN_degree)
    X_Node = torch.tensor(node_inds).to(device)
    X_Neis = torch.tensor(node_neis).to(device)
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
            y1_ground[data_id*num_frames+image_id, :] = -dqu[2, :] / points_3D[2, :]  #Ground truth
            y2_ground[data_id*num_frames+image_id, :] = -dqv[2, :] / points_3D[2, :]
            points_3D_all[data_id*num_frames+image_id, [0,1,2],:] =  points_3D

###################################################################################################################################################
    #latent_space = torch.zeros(2, num_point_per_frame, requires_grad= True)
    latent_space_const = torch.zeros(1, num_point_per_frame).to(device)
    #latent_space = torch.randn(2, num_point_per_frame, requires_grad=True, device=device)
    #parameters_to_optimiza=[{'params': latent_space}]
    #############################
    #parameters_to_optimiza.append({"params":shape_partial_derivate[0].parameters()})
    parameters_to_optimiza=[{'params': shape_partial_derivate[0].parameters()}]
    parameters_to_optimiza.append({"params":shape_partial_derivate[1].parameters()})
    #for frame_idx in range(num_frames):
    #    parameters_to_optimiza.append({"params":shape_partial_derivate[frame_idx].parameters()})
    #optimizer = torch.optim.Rprop(parameters_to_optimiza, lr=0.0001, step_sizes=(1e-10, 50))
    optimizer = torch.optim.Adam(parameters_to_optimiza, lr=0.0001)
    scaler = GradScaler()  # AMP
    loss = torch.nn.MSELoss()#torch.nn.L1Loss()
    batch_size = 2 #12
    num_train = num_data * num_frames
    for i in range(num_iterations):
        ## Training
        shape_partial_derivate[0].train()
        shape_partial_derivate[1].train()
        # y1_predict = torch.zeros(num_data * num_frames, num_point_per_frame)
        # y2_predict = torch.zeros(num_data * num_frames, num_point_per_frame)
        #output = torch.zeros(1).to(device)
        for train_batch_id in range(num_train // batch_size):
            optimizer.zero_grad()
            #torch.cuda.empty_cache()
            with autocast():  # AMP
                y_result = shape_partial_derivate[0].forward(points_3D_all[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:,:])
                y_result_latent = shape_partial_derivate[1].forward(points_3D_all[train_batch_id * batch_size:(train_batch_id + 1) * batch_size, :, :])
                loss1 = loss(y_result, y1_ground[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:])
                loss2 = loss(y_result_latent, y2_ground[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:])
                output = loss1  + loss2
            output.backward()  # retain_graph=True
            optimizer.step()
        shape_partial_derivate[0].eval()
        shape_partial_derivate[1].eval()


        '''for i in range(num_iterations):
        ## Training
        这是一个“非 mini-batch”的循环方案。逐 data_id 和 image_id 计算 points_3D、points_3D_latent，走一次大前向并用 latent_space 做除法后构造损失，再 backward。
        特点：显存占用大、速度慢、图很复杂，且 retain_graph 的使用在这个思路里更加危险。
        optimizer.zero_grad()
        #y1_predict = torch.zeros(num_data * num_frames, num_point_per_frame)
        #y2_predict = torch.zeros(num_data * num_frames, num_point_per_frame)
        output=torch.zeros(1).to(device)
        for data_id in range(num_data):
            for image_id in range(num_frames):
                points_3D = normilized_point[image_id, [0, 1, 2], :] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
                points_3D_latent = (normilized_point[image_id, [0, 1, 2], :] + torch.cat((latent_space[[0, 1],:], latent_space_const),0)) * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
                #y_result = shape_partial_derivate.forward(X_Node, X_Neis, points_3D)
                y_result = shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D, normilized_point[image_id, [0, 1, ], :]), 0))

                y_result_latent = shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D_latent, normilized_point[image_id, [0, 1, ], :]), 0))#Way 2
                ###################
                #y_result = shape_partial_derivate[image_id].forward(X_Node, X_Neis, torch.cat((points_3D, normilized_point[image_id, [0, 1, ], :]), 0))
                #y_result_latent = shape_partial_derivate[image_id].forward(X_Node, X_Neis, torch.cat((points_3D_latent, normilized_point[image_id, [0, 1, ], :]), 0))#Way 2
                ###################
                # y_result_latent = shape_partial_derivate.forward(X_Node, X_Neis, torch.cat((points_3D + points_3D_latent, normilized_point[image_id, [0, 1, ], :]), 0))#Way 1
                #y_result_latent = shape_partial_derivate.forward(X_Node, X_Neis, torch.cat((points_3D + points_3D_latent, normilized_point[image_id, [0, 1, ], :]), 0))#Way 3

                #y_result = shape_partial_derivate.forward(torch.flatten(torch.cat((points_3D, normilized_point[image_id, [0, 1, ], :]), 0)))
                #good y_result = shape_partial_derivate[image_id].forward(torch.flatten(torch.transpose(torch.cat((points_3D, normilized_point[image_id, [0, 1, ], :]), 0), 0, 1)))
                #good y_result_latent = shape_partial_derivate[image_id].forward(torch.flatten(torch.transpose(torch.cat((points_3D_latent, normilized_point[image_id, [0, 1, ], :]), 0), 0, 1)))
                #y1_predict[data_id*num_frames+image_id, :] = y_result[:num_point_per_frame]
                #y2_predict[data_id*num_frames+image_id, :] = y_result[num_point_per_frame:]
                loss1 = loss((y_result_latent[:num_point_per_frame]-y_result[:num_point_per_frame])/latent_space[0,:], y1_ground[data_id*num_frames+image_id, :])
                loss2 = loss((y_result_latent[num_point_per_frame:]-y_result[num_point_per_frame:])/latent_space[1,:], y2_ground[data_id*num_frames+image_id, :])
                output = loss1  + output + loss2

        #loss2 = loss(y2_predict, y2_ground)

        output.backward(retain_graph=True)  # retain_graph=True
        optimizer.step()'''



        #y1_predict_ref = shape_partial_derivate[0].forward(points_3D_all[0:8, :, :])
        #y2_predict_ref = shape_partial_derivate[1].forward(points_3D_all[0:8, :, :])
        #accuracy = torch.sum(((y1_predict_ref - y1_ground[0:8,:])**2+(y2_predict_ref - y2_ground[0:8,:])**2)).detach().to("cpu").numpy()

        '''        accuracy = []
        作用：基于“参考前向”计算 y1/y2 的预测误差，作为准确度输出。
        for data_id in range(num_data):
            for image_id in range(num_frames):
                points_3D_ref = normilized_point[image_id, [0, 1, 2], :] * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
                points_3D_ref_latent = (normilized_point[image_id, [0, 1, 2], :] + torch.cat((latent_space[[0, 1],:], latent_space_const),0)) * random_depth_data[data_id * num_frames + image_id, :].repeat(3, 1)
                y_result_ref = shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D_ref, normilized_point[image_id, [0, 1], :]), 0))
                y_result_ref_latent = shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D_ref_latent, normilized_point[image_id, [0, 1], :]), 0))#Way 1
                ###################
                #y_result_ref = shape_partial_derivate[image_id].forward(X_Node, X_Neis, torch.cat((points_3D_ref, normilized_point[image_id, [0, 1], :]), 0))
                #y_result_ref_latent = shape_partial_derivate[image_id].forward(X_Node, X_Neis, torch.cat((points_3D_ref_latent, normilized_point[image_id, [0, 1], :]), 0))#Way 1
                ###############
                #y_result_ref_latent = shape_partial_derivate.forward(X_Node, X_Neis, torch.cat((points_3D_ref_latent, normilized_point[0, [0, 1], :]), 0))#Way 2
                #y_result_ref_latent = shape_partial_derivate.forward(X_Node, X_Neis, torch.cat((points_3D_ref+points_3D_ref_latent, normilized_point[0, [0, 1], :]), 0))#Way 2
                #y_result_ref = shape_partial_derivate.forward(X_Node, X_Neis, points_3D_ref)
                #y_result_ref = shape_partial_derivate.forward( torch.flatten(points_3D_ref))
                #y_result_ref = shape_partial_derivate[image_id].forward(torch.flatten(torch.transpose(torch.cat((points_3D_ref, normilized_point[image_id, [0, 1, ], :]), 0), 0, 1)))
                #y_result_ref_latent = shape_partial_derivate[image_id].forward(torch.flatten(torch.transpose(torch.cat((points_3D_ref_latent, normilized_point[image_id, [0, 1, ], :]), 0), 0, 1)))
                y1_predict_ref = (y_result_ref_latent[:num_point_per_frame]-y_result_ref[:num_point_per_frame])/latent_space[0,:]
                y2_predict_ref = (y_result_ref_latent[num_point_per_frame:]-y_result_ref[num_point_per_frame:])/latent_space[1,:]
                accuracy.append(torch.sum(((y1_predict_ref - y1_ground[data_id*num_frames+image_id,:])**2+(y2_predict_ref - y2_ground[data_id*num_frames+image_id,:])**2)).detach().to("cpu").numpy())
                '''

        if i % 3 == 2:  # print every 3 iterations
            #print('[%5d, %5d] loss: %.6f' % (i + 1, num_iterations, output.data))
            print('[%5d, %5d] loss: %.6f' % (i + 1, num_iterations, float(output.detach().cpu())))
            #print(*accuracy, sep = ", ")
            print("\n")

    '''
    作用：基于 Initial_shape 的 points_3D，重新跑一次拟合得到 y1_ground_accuracy/y2_ground_accuracy，再做 forward 得到评估。
    y1_ground_accuracy = torch.zeros(num_frames, num_point_per_frame).to(device)
    y2_ground_accuracy = torch.zeros(num_frames, num_point_per_frame).to(device)
    #y_result_accuracy = torch.zeros(num_frames, num_point_per_frame).to(device)
    #y_result_latent_accuracy = torch.zeros(num_frames, num_point_per_frame).to(device)
    points_3D = torch.zeros(num_frames, 3, num_point_per_frame).to(device)
    for image_id in range(num_frames):
        points_3D[image_id, :, :] = torch.unsqueeze(normilized_point[image_id, [0, 1, 2], :] * torch.tensor(Initial_shape[image_id, :]).to(device).repeat(3, 1), 0)
        uv = matlab.double(normilized_point[image_id, [0, 1], :].tolist())
        fit_result = m.fit_python(uv, matlab.double(points_3D[image_id, :, :].tolist()), uv, nargout=6)
        dqu = torch.tensor(np.array(fit_result[1])).to(device)
        dqv = torch.tensor(np.array(fit_result[2])).to(device)
        y1_ground_accuracy[image_id, :] = dqu[2, :] / points_3D[image_id, 2, :]  # Ground truth
        y2_ground_accuracy[image_id, :] = dqv[2, :] / points_3D[image_id, 2, :]
    y_result_accuracy = shape_partial_derivate[0].forward(points_3D)
    y_result_latent_accuracy = shape_partial_derivate[1].forward(points_3D)  # Way 1'''
    print('accuracy: %.8f (u)  %.8f (v)' % (torch.sqrt(loss1), torch.sqrt(loss2)))
    print('accuracy: %.8f (u)  %.8f (v)' % (torch.sqrt(loss1)/torch.mean(torch.abs(y_result)),
                                        torch.sqrt(loss2)/torch.mean(torch.abs(y_result_latent))))

    # 增加表1口径的 MAE/mean(|.|) 打印
    with torch.no_grad():
        mae_u = torch.mean(torch.abs(y_result - y1_ground[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:]))
        mae_v = torch.mean(torch.abs(y_result_latent - y2_ground[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:]))
        meanabs_u = torch.mean(torch.abs(y1_ground[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:]))
        meanabs_v = torch.mean(torch.abs(y2_ground[train_batch_id*batch_size:(train_batch_id+1)*batch_size,:]))
        print('MAE: %.8f (u)  %.8f (v)' % (mae_u, mae_v))
        print('MAE/meanabs: %.8f (u)  %.8f (v)' % (mae_u/meanabs_u, mae_v/meanabs_v))

    return  shape_partial_derivate, [] #, latent_space #shape_weights