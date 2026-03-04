from Dataset.loss_parameter import loss_functions_params
import numpy as np
import scipy.io
import trimesh
import math
import matlab.engine
#from NRSfM_core.model_develop import learning_model
from NRSfM_core.KNN_graph import Graph_distance
from ctypes import cdll
import open3d as o3d
#import open3d.ml.torch as ml3d
import torch
#from pytorch3d.structures import Meshes
#from pytorch3d.ops import knn_points
from NRSfM_core.shape_decoder import ShapeDecoder
from NRSfM_core.class_autograd import ChamferFunction
#from pytorch3d.loss import mesh_laplacian_smoothing
from torch.autograd.functional import jacobian

class NRSfMLoss:

    def __init__(self, scene_normalized, num_points, J, m, device, degree, normilized_point):
        self.num_frames = scene_normalized.shape[0]
        self.num_point_per_frame=scene_normalized.shape[2]
        self.normilized_point_batched=np.zeros(shape=(self.num_frames,3,self.num_point_per_frame), dtype=np.float32)
        for frame_idx in range(self.num_frames):
            self.normilized_point_batched[frame_idx, 0, :] = scene_normalized[frame_idx, 0, :]
            self.normilized_point_batched[frame_idx, 1, :] = scene_normalized[frame_idx, 1, :]
            self.normilized_point_batched[frame_idx, 2, :] = np.ones(scene_normalized.shape[2])
        self.num_points = num_points
        self.normilized_point = torch.tensor(self.normilized_point_batched).to(device)
        self.device = device
        if self.num_point_per_frame <= 200:
            self.omega_CC_and_MC = float(self.num_point_per_frame) ** 2 / 200. / 200.
        else:
            if self.num_point_per_frame > 300:
                self.omega_CC_and_MC = 3 * (1 - math.exp(- (float(self.num_point_per_frame)-300.) / 100.))
            else:
                self.omega_CC_and_MC = 1
        self.omega_ARAP = 1
        self.m = m
        if J!= 0.:
            self.J=J
        else:
            print("need to compute image warp")
            exit()
        self.k = degree  # Nearest 20 points
        #self.knnrst = knn_points(torch.unsqueeze(self.normilized_point_batched[0,:,:],0),torch.unsqueeze(self.normilized_point_batched[0,:,:],0), None, None, self.k+1)
        self.ID = Graph_distance(self.normilized_point_batched[0, [0,1], :], self.k )
        if self.num_point_per_frame <= 500:
            self.approximate_weight = 3
        else:
            self.approximate_weight = 1
        self.node_inds = np.repeat(np.arange(num_points), degree)
        self.node_neis = self.ID.flatten()

        self.tJ = {
            'du_dubar': torch.tensor(self.J.dx1_dy1, device=self.device),
            'du_dvbar': torch.tensor(self.J.dx1_dy2, device=self.device),
            'dv_dubar': torch.tensor(self.J.dx2_dy1, device=self.device),
            'dv_dvbar': torch.tensor(self.J.dx2_dy2, device=self.device),
            'dubar_du': torch.tensor(self.J.dy1_dx1, device=self.device),
            'dubar_dv': torch.tensor(self.J.dy1_dx2, device=self.device),
            'dvbar_du': torch.tensor(self.J.dy2_dx1, device=self.device),
            'dvbar_dv': torch.tensor(self.J.dy2_dx2, device=self.device),
            'ddu_dudvbar': torch.tensor(self.J.ddx1_dxdy, device=self.device),
            'ddv_dudvbar': torch.tensor(self.J.ddx2_dxdy, device=self.device),
        }
        # 预缓存归一化坐标
        self._u0 = torch.tensor(self.normilized_point_batched[0, 0, :], device=self.device)
        self._v0 = torch.tensor(self.normilized_point_batched[0, 1, :], device=self.device)
        self._u_all = torch.tensor(self.normilized_point_batched[:, 0, :], device=self.device)
        self._v_all = torch.tensor(self.normilized_point_batched[:, 1, :], device=self.device)
        # 预转换 uv 为 matlab.double
        import matlab
        self._uv_matlab = []
        for f in range(self.num_frames):
            self._uv_matlab.append(
                matlab.double(self.normilized_point_batched[f, [0, 1], :].tolist())
            )



    '''def loss_subterm_connection_and_metric_tensor1(self, depth):
        points_3D = torch.from_numpy(self.normilized_point_batched) * depth.repeat(1, 3, 1)
        points_3D_1=points_3D.cpu().detach().numpy()
        loss_subterm_connection_value=0
        y1 = np.array([[0.0] * self.num_point_per_frame] * self.num_frames)
        y2 = np.array([[0.0] * self.num_point_per_frame] * self.num_frames)
        for frame_idx in range(self.num_frames):
            uv = matlab.double(self.normilized_point_batched[frame_idx, [0,1], :].tolist())
            points_3d = matlab.double(points_3D_1[frame_idx, :, :].tolist())
            fit_result = self.m.fit_python(uv, points_3d, uv, nargout=6)
            dqu = np.array(fit_result[1])
            dqv = np.array(fit_result[2])
            y1[frame_idx,:] = np.array(fit_result[1])[2,:]/ points_3D_1[frame_idx, 2, :]
            y2[frame_idx,:] = np.array(fit_result[2])[2,:]/ points_3D_1[frame_idx, 2, :]

        for frame_idx in range(self.num_frames-1):
            du_dubar = self.J.dx1_dy1
            du_dvbar = self.J.dx1_dy2
            dv_dubar = self.J.dx2_dy1
            dv_dvbar = self.J.dx2_dy2
            ddu_ddubar = self.J.ddx1_ddy1
            ddu_ddvbar = self.J.ddx1_ddy2
            ddv_ddubar = self.J.ddx2_ddy1
            ddv_ddvbar = self.J.ddx2_ddy2
            ddu_dudvbar = self.J.ddx1_dxdy
            ddv_dudvbar = self.J.ddx2_dxdy
            u = self.normilized_point_batched[0, 0, :]
            v = self.normilized_point_batched[0, 1, :]
            u_bar = self.normilized_point_batched[frame_idx+1, 0, :]
            v_bar = self.normilized_point_batched[frame_idx+1, 1, :]
            ## connection invariance
            eq1=du_dubar[frame_idx,:]*y1[0,:]+dv_dubar[frame_idx,:]*y2[0,:]+dv_dubar[frame_idx,:]*ddu_dudvbar[frame_idx,:]+dv_dvbar[frame_idx,:]*ddv_dudvbar[frame_idx,:]-y1[frame_idx+1,:]
            eq2=du_dvbar[frame_idx,:]*y1[0,:]+dv_dvbar[frame_idx,:]*y2[0,:]+du_dubar[frame_idx,:]*ddu_dudvbar[frame_idx,:]+du_dvbar[frame_idx,:]*ddv_dudvbar[frame_idx,:]-y2[frame_idx+1,:]
            ## metric tensor invariance
            G11 = y1[0, :] * y1[0, :] + (y1[0, :] * u + 1) * (y1[0, :] * u + 1) + (y1[0, :] * v) * (y1[0, :] * v)
            G12 = y1[0, :] * y2[0, :] * (1 + u * u + v * v) + y2[0, :] * u + y1[0, :] * v
            G22 = y2[0, :] * y2[0, :] + (y1[0, :] * v + 1) * (y1[0, :] * v + 1) + (y2[0, :] * u) * (y2[0, :] * u)
            G21 = y1[0, :] * y2[0, :] * (1 + u * u + v * v) + y2[0, :] * u + y1[0, :] * v
            G11_bar = y1[frame_idx+1, :] * y1[frame_idx+1, :] + (y1[frame_idx+1, :] * u_bar + 1) * (y1[frame_idx+1, :] * u_bar + 1) + (y1[frame_idx+1, :] * v_bar) * (y1[frame_idx+1, :] * v_bar)
            G12_bar = y1[frame_idx+1, :] * y2[frame_idx+1, :] * (1 + u_bar * u_bar + v_bar * v_bar) + y2[frame_idx+1, :] * u_bar + y1[frame_idx+1, :] * v_bar
            G22_bar = y2[frame_idx+1, :] * y2[frame_idx+1, :] + (y1[frame_idx+1, :] * v_bar + 1) * (y1[frame_idx+1, :] * v_bar + 1) + (y2[frame_idx+1, :] * u_bar) * (y2[frame_idx+1, :] * u_bar)
            G21_bar = y1[frame_idx+1, :] * y2[frame_idx+1, :] * (1 + u_bar * u_bar + v_bar * v_bar) + y2[frame_idx+1, :] * u_bar + y1[frame_idx+1, :] * v_bar
            eq3=du_dubar[frame_idx,:]*du_dubar[frame_idx,:]*G11*G12_bar+du_dubar[frame_idx,:]*dv_dubar[frame_idx,:]*G12*G12_bar+dv_dubar[frame_idx,:]*du_dubar[frame_idx,:]*G21*G12_bar+dv_dubar[frame_idx,:]*dv_dubar[frame_idx,:]*G22*G12_bar-du_dubar[frame_idx,:]*du_dvbar[frame_idx,:]*G11*G11_bar-du_dubar[frame_idx,:]*dv_dvbar[frame_idx,:]*G12*G11_bar-dv_dubar[frame_idx, :] * du_dvbar[frame_idx, :] * G21 * G11_bar -dv_dubar[frame_idx, :] * dv_dvbar[frame_idx, :] * G22 * G11_bar
            eq4=du_dubar[frame_idx,:]*du_dubar[frame_idx,:]*G11*G22_bar+du_dubar[frame_idx,:]*dv_dubar[frame_idx,:]*G12*G22_bar+dv_dubar[frame_idx,:]*du_dubar[frame_idx,:]*G21*G22_bar+dv_dubar[frame_idx,:]*dv_dubar[frame_idx,:]*G22*G22_bar-du_dvbar[frame_idx,:]*du_dvbar[frame_idx,:]*G11*G11_bar-du_dvbar[frame_idx,:]*dv_dvbar[frame_idx,:]*G12*G11_bar-dv_dvbar[frame_idx, :] * du_dvbar[frame_idx, :] * G21 * G11_bar -dv_dvbar[frame_idx, :] * dv_dvbar[frame_idx, :] * G22 * G11_bar
            loss_subterm_connection_value = loss_subterm_connection_value + np.sum(eq1 ** 2, axis=0)+np.sum(eq2 ** 2, axis=0)+np.sum(eq3 ** 2, axis=0)+np.sum(eq4 ** 2, axis=0)

        return loss_subterm_connection_value'''

    def loss_subterm_connection_and_metric_tensor(self, depth, Shape_partial_derivate, model_derivation, frame_indices=None):
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))
        
        loss_subterm_connection_value_1 = torch.zeros(1, 1).to(self.device)
        loss_subterm_connection_value_2 = torch.zeros(1, 1).to(self.device)
        y1 = torch.zeros(self.num_frames, self.num_point_per_frame).to(self.device)
        y2 = torch.zeros(self.num_frames, self.num_point_per_frame).to(self.device)
        latent_space_const = torch.zeros(1, self.num_point_per_frame).to(self.device)
        m = self.m
        
        # Need to compute y for frame 0 (reference) AND specified batch frames
        compute_indices = sorted(list(set(frame_indices) | {0}))

        if model_derivation:
            for frame_idx in compute_indices:
                Depth = depth[frame_idx, :, :]
                # Way 1 partial derivative
                Depth1 = Depth.detach() #Dete, if only using partial derivative, it will use lf-write backward  y_result_mark = ChamferFunction.apply(Depth, self.normilized_point_batched[frame_idx, :, :], m)
                y_result_mark = ChamferFunction.apply(Depth1, self.normilized_point_batched[frame_idx, :, :], m, self.device)
                #y1[frame_idx, :] = y_result_mark[:self.num_point_per_frame]
                #y2[frame_idx, :] = y_result_mark[self.num_point_per_frame:]
                # Way 2 network approximation
                points_3D = torch.from_numpy(self.normilized_point_batched[frame_idx, :, :]).to(self.device) * Depth.repeat(3, 1)
                X_Node = torch.tensor(self.node_inds).to(self.device)
                X_Neis = torch.tensor(self.node_neis).to(self.device)
                #y_result = Shape_partial_derivate.forward(X_Node, X_Neis, points_3D)
                #y_result = Shape_partial_derivate[frame_idx].forward(torch.flatten(torch.transpose(torch.cat((points_3D, self.normilized_point[frame_idx, [0, 1, ], :]), 0), 0, 1)))
                points_3D_latent = (self.normilized_point[frame_idx, [0, 1, 2], :] + torch.cat((model_derivation[[0, 1], :], latent_space_const), 0)) * Depth.repeat(3, 1)
                #y_result_latent = Shape_partial_derivate[frame_idx].forward(torch.flatten(torch.transpose(torch.cat((points_3D_latent, self.normilized_point[frame_idx, [0, 1, ], :]), 0), 0, 1)))
                y_result = Shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D, self.normilized_point[frame_idx, [0, 1], :]), 0))
                y_result_latent = Shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D_latent, self.normilized_point[frame_idx, [0, 1], :]), 0))  # Way 1
                ####################
                #y_result = Shape_partial_derivate[frame_idx].forward(X_Node, X_Neis, torch.cat((points_3D, self.normilized_point[frame_idx, [0, 1], :]), 0))
                #y_result_latent = Shape_partial_derivate[frame_idx].forward(X_Node, X_Neis, torch.cat((points_3D_latent, self.normilized_point[frame_idx, [0, 1], :]), 0))  # Way 1
                ################
                # y1_predict[data_id*num_frames+image_id, :] = y_result[:num_point_per_frame]
                # y2_predict[data_id*num_frames+image_id, :] = y_result[num_point_per_frame:]
                #loss1 = loss((y_result_latent[:num_point_per_frame] - y_result[:num_point_per_frame]) / latent_space[0, :], y1_ground[data_id * num_frames + image_id, :])
                #loss2 = loss((y_result_latent[num_point_per_frame:] - y_result[num_point_per_frame:]) / latent_space[1, :], y2_ground[data_id * num_frames + image_id, :])
                y_result_final = torch.cat(((y_result_latent[:self.num_point_per_frame] - y_result[:self.num_point_per_frame]) / model_derivation[0, :], (y_result_latent[self.num_point_per_frame:] - y_result[self.num_point_per_frame:]) / model_derivation[1, :]), 0)

                y1[frame_idx, :] = y_result_final[:self.num_point_per_frame]
                y2[frame_idx, :] = y_result_final[self.num_point_per_frame:]

                #y1[frame_idx, :] = y_result_mark[:self.num_point_per_frame]
                #y2[frame_idx, :] = y_result_mark[self.num_point_per_frame:]

                # Only add connection loss for frames in current batch
                if frame_idx in frame_indices:
                    loss_subterm_connection_value_1 = loss_subterm_connection_value_1 + self.approximate_weight * torch.sum(torch.square(y_result_final-y_result_mark))
        else:
            '''            
                for frame_idx in range(self.num_frames):
                points_3D = torch.zeros(self.num_frames + 1, 3, self.num_point_per_frame).to(self.device)
                y_result_mark = torch.zeros(self.num_frames + 1, self.num_point_per_frame * 2).to(self.device)
                Depth = depth[frame_idx, :, :]
                Depth1 = Depth.detach() #Dete, if only using partial derivative, it will use lf-write backward  y_result_mark = ChamferFunction.apply(Depth, self.normilized_point_batched[frame_idx, :, :], m)
                y_result_mark = ChamferFunction.apply(Depth1, self.normilized_point_batched[frame_idx, :, :], m, self.device)
                points_3D = torch.from_numpy(self.normilized_point_batched[frame_idx, :, :]).to(self.device) * Depth.repeat(3, 1)
                y_result = Shape_partial_derivate[0].forward(torch.unsqueeze(points_3D, 0))
                y_result_latent = Shape_partial_derivate[1].forward(torch.unsqueeze(points_3D, 0))
                y_result_final = torch.squeeze(torch.cat((y_result, y_result_latent), 1))
                y1[frame_idx, :] = y_result
                y2[frame_idx, :] = y_result_latent
                loss_subterm_connection_value_1 = loss_subterm_connection_value_1 + self.approximate_weight * torch.sum(torch.square(y_result_final-y_result_mark))
            '''
            points_3D = torch.zeros(self.num_frames, 3, self.num_point_per_frame).to(self.device)
            y_result_mark = torch.zeros(self.num_frames, self.num_point_per_frame * 2).to(self.device)
            for frame_idx in compute_indices:
                Depth = depth[frame_idx, :, :]
                Depth1 = Depth.detach()
                # 内联 ChamferFunction.forward（depth已detach，backward不会被调用）
                pts3d = self.normilized_point[frame_idx] * Depth1.repeat(3, 1)
                fit_result = m.fit_python(
                    self._uv_matlab[frame_idx],
                    matlab.double(pts3d.detach().cpu().numpy().tolist()),
                    self._uv_matlab[frame_idx],
                    nargout=6
                )
                dqu = torch.tensor(fit_result[1], device=self.device)
                dqv = torch.tensor(fit_result[2], device=self.device)
                y1_mark = -dqu[2, :] / pts3d[2, :]
                y2_mark = -dqv[2, :] / pts3d[2, :]
                y_result_mark[frame_idx, :] = torch.cat((y1_mark, y2_mark), 0)
                points_3D[frame_idx, :, :] = self.normilized_point[frame_idx] * Depth.repeat(3, 1)
            
            # Use only compute_indices for GNN to save memory
            y_result = Shape_partial_derivate[0].forward(points_3D[compute_indices])
            y_result_latent = Shape_partial_derivate[1].forward(points_3D[compute_indices])
            y_result_final_all = torch.squeeze(torch.cat((y_result, y_result_latent), 1))
            
            # Map back to full y1, y2
            for i, idx in enumerate(compute_indices):
                y1[idx, :] = y_result[i]
                y2[idx, :] = y_result_latent[i]
                if idx in frame_indices:
                    loss_subterm_connection_value_1 = loss_subterm_connection_value_1 + self.approximate_weight * torch.sum(torch.square(y_result_final_all[i] - y_result_mark[idx]))

        for frame_idx_m1 in range(self.num_frames-1):
            frame_idx = frame_idx_m1 + 1
            if frame_idx not in frame_indices:
                continue

            du_dubar = self.tJ['du_dubar']
            du_dvbar = self.tJ['du_dvbar']
            dv_dubar = self.tJ['dv_dubar']
            dv_dvbar = self.tJ['dv_dvbar']
            dubar_du = self.tJ['dubar_du']
            dubar_dv = self.tJ['dubar_dv']
            dvbar_du = self.tJ['dvbar_du']
            dvbar_dv = self.tJ['dvbar_dv']
            ddu_dudvbar = self.tJ['ddu_dudvbar']
            ddv_dudvbar = self.tJ['ddv_dudvbar']
            u = self._u0
            v = self._v0
            u_bar = torch.tensor(self.normilized_point_batched[frame_idx, 0, :], device=self.device)
            v_bar = torch.tensor(self.normilized_point_batched[frame_idx, 1, :], device=self.device)



            ## connection invariance
            eq1=du_dubar[frame_idx-1,:]*y1[0,:]+dv_dubar[frame_idx-1,:]*y2[0,:]-dvbar_du[frame_idx-1,:]*ddu_dudvbar[frame_idx-1,:]-dvbar_dv[frame_idx-1,:]*ddv_dudvbar[frame_idx-1,:]-y1[frame_idx,:]
            eq2=du_dvbar[frame_idx-1,:]*y1[0,:]+dv_dvbar[frame_idx-1,:]*y2[0,:]-dubar_du[frame_idx-1,:]*ddu_dudvbar[frame_idx-1,:]-dubar_dv[frame_idx-1,:]*ddv_dudvbar[frame_idx-1,:]-y2[frame_idx,:]
            ## metric tensor invariance
            e1 = u*u+ v*v + 1
            e1_bar = u_bar*u_bar + v_bar*v_bar + 1
            b1 = u*u + 1
            b1_bar = u_bar*u_bar+1
            b2 = v*v + 1
            b2_bar = v_bar*v_bar+1
            G11 = (y1[0, :] * u-1)*(y1[0, :] * u-1)+b2*y1[0, :]*y1[0, :]
            G12 = e1*y1[0, :] * y2[0, :] -y2[0, :] * u-y1[0, :] * v
            G22 = (y2[0, :] * v-1)*(y2[0, :] * v-1)+b1*y2[0, :]*y2[0, :]
            G21 = G12
            G11_bar = (y1[frame_idx, :]*u_bar-1)*(y1[frame_idx, :]*u_bar-1)+b2_bar*y1[frame_idx, :]*y1[frame_idx, :]
            G12_bar = e1_bar*y1[frame_idx, :]*y2[frame_idx, :]-y2[frame_idx, :]*u_bar-y1[frame_idx, :]*v_bar
            G22_bar = (y2[frame_idx, :]*v_bar-1)*(y2[frame_idx, :]*v_bar-1)+b1_bar*y2[frame_idx, :]*y2[frame_idx, :]
            G21_bar = G12_bar
            eq3=du_dubar[frame_idx-1,:]*du_dubar[frame_idx-1,:]*G11*G12_bar+du_dubar[frame_idx-1,:]*dv_dubar[frame_idx-1,:]*G12*G12_bar+dv_dubar[frame_idx-1,:]*du_dubar[frame_idx-1,:]*G21*G12_bar+dv_dubar[frame_idx-1,:]*dv_dubar[frame_idx-1,:]*G22*G12_bar-du_dubar[frame_idx-1,:]*du_dvbar[frame_idx-1,:]*G11*G11_bar-du_dubar[frame_idx-1,:]*dv_dvbar[frame_idx-1,:]*G12*G11_bar-dv_dubar[frame_idx-1, :] * du_dvbar[frame_idx-1, :] * G21 * G11_bar -dv_dubar[frame_idx-1, :] * dv_dvbar[frame_idx-1, :] * G22 * G11_bar
            eq4=du_dubar[frame_idx-1,:]*du_dubar[frame_idx-1,:]*G11*G22_bar+du_dubar[frame_idx-1,:]*dv_dubar[frame_idx-1,:]*G12*G22_bar+dv_dubar[frame_idx-1,:]*du_dubar[frame_idx-1,:]*G21*G22_bar+dv_dubar[frame_idx-1,:]*dv_dubar[frame_idx-1,:]*G22*G22_bar-du_dvbar[frame_idx-1,:]*du_dvbar[frame_idx-1,:]*G11*G11_bar-du_dvbar[frame_idx-1,:]*dv_dvbar[frame_idx-1,:]*G12*G11_bar-dv_dvbar[frame_idx-1, :] * du_dvbar[frame_idx-1, :] * G21 * G11_bar -dv_dvbar[frame_idx-1, :] * dv_dvbar[frame_idx-1, :] * G22 * G11_bar
            loss_subterm_connection_value_2 = loss_subterm_connection_value_2 + torch.sum(torch.square(eq1))+torch.sum(torch.square(eq2))+torch.sum(torch.square(eq3))+torch.sum(torch.square(eq4))

        loss_subterm_connection_value = loss_subterm_connection_value_1 + loss_subterm_connection_value_2
        print('Partial loss 1:  %.3f (Conection)  %.3f (Approximation)' % (loss_subterm_connection_value_2, loss_subterm_connection_value_1))
        return loss_subterm_connection_value

    '''def loss_subterm_distance_invariance(self, depth):
        points_3D = torch.from_numpy(self.normilized_point_batched).to(self.device) * depth.repeat(1, 3, 1)
        loss_subterm_distance_value = torch.zeros(1, 1).to(self.device)

        # Way 1 Get closed to mean value
        for point_idx in range(self.num_points):
            for frame_idx in range(self.num_frames - 1):
                point_difference1 = torch.unsqueeze(points_3D[frame_idx, :, point_idx],1).repeat(1,self.k)- points_3D[frame_idx, :, self.ID[point_idx, :]]
                point_difference2 = torch.unsqueeze(points_3D[frame_idx + 1, :, point_idx],1).repeat(1,self.k)- points_3D[frame_idx+1, :, self.ID[point_idx, :]]
                dist0 = torch.sqrt(torch.sum(point_difference1 ** 2, 0))
                dist1 = torch.sqrt(torch.sum(point_difference2 ** 2, 0))
                loss_subterm_distance_value = loss_subterm_distance_value + torch.sum((dist1 - dist0) ** 2)

        # Way 2 Gets close to largest value with 3D points
        point_difference1 = torch.zeros(self.num_frames, self.num_points, self.k, 3).to(self.device)
        max_value = torch.zeros(1, self.num_points, self.k).to(self.device)
        dist0 = torch.zeros(self.num_frames, self.num_points, self.k).to(self.device)
        for point_idx in range(self.num_points):
            for adjuest in range(self.k):
                for frame_idx in range(self.num_frames):
                    point_difference1[frame_idx, point_idx, adjuest,:] = torch.transpose(torch.unsqueeze(points_3D[frame_idx, :, point_idx], 1),1,0) - points_3D[frame_idx, :, self.ID[point_idx, adjuest]]
                    dist0[frame_idx, point_idx, adjuest] = torch.sqrt(torch.sum(point_difference1[frame_idx, point_idx, adjuest, :] ** 2, 0))
                max_value[0, point_idx, adjuest] = torch.max(dist0[:, point_idx, adjuest])
        loss_subterm_distance_value = loss_subterm_distance_value + torch.sum((dist0 - max_value.repeat(self.num_frames, 1, 1))**2)


        # Way 2 Gets close to largest value with 3D points (some bugs)
        point_difference1 = torch.zeros(self.num_frames, self.num_points, self.k, 3)
        max_value = torch.zeros(1, self.num_points, self.k)
        dist0 = torch.zeros(self.num_frames, self.num_points, self.k)
        for point_idx in range(self.num_points):
            for adjuest in range(self.k):
                for frame_idx in range(self.num_frames):
                    point_difference1[frame_idx, point_idx, adjuest,:] = torch.transpose(torch.unsqueeze(points_3D[frame_idx, :, point_idx], 1),1,0) - points_3D[frame_idx, :, self.ID[point_idx, adjuest]]
                    dist0[frame_idx, point_idx, adjuest] = torch.sqrt(torch.sum(point_difference1[frame_idx, point_idx, adjuest, :] ** 2, 0))
                max_value[0, point_idx, adjuest] = torch.max(dist0[:, point_idx, adjuest])
        loss_subterm_distance_value = loss_subterm_distance_value + torch.sum((dist0 - max_value.repeat(self.num_frames, 1, 1))**2

        # Way 2 Gets close to largest value with 3D points
                for point_idx in range(self.num_points):
            for adjuest in range(self.k):
                dist0 = torch.zeros(1, self.num_frames).to(self.device)
                for frame_idx in range(self.num_frames):
                    point_difference1 = points_3D[frame_idx, :, point_idx]- points_3D[frame_idx, :, self.ID[point_idx, adjuest]]
                    dist0[0, frame_idx] = torch.sqrt(torch.sum(point_difference1 ** 2))
                max_value = torch.max(dist0)
                loss_subterm_distance_value = loss_subterm_distance_value + torch.sum((dist0 - max_value.repeat(1, self.num_frames))**2)
        
        return loss_subterm_distance_value'''

    def loss_subterm_distance_invariance(self, depth, frame_indices=None):
        """
        Vectorized implementation for both ARAP and AMAP.
        depth: [F, 1, N] or [F, N]
        """
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        # 1. 构造 3D 点云 [F_batch, 3, N]
        if depth.dim() == 2:
            depth_batch = depth[frame_indices].unsqueeze(1) # [F_batch, 1, N]
        else:
            depth_batch = depth[frame_indices]
        
        # self.normilized_point: [F, 3, N]
        points_3D = self.normilized_point[frame_indices] * depth_batch.repeat(1, 3, 1)
        
        F, _, N = points_3D.shape
        K = self.k
        
        # 2. 获取邻居索引
        if isinstance(self.ID, np.ndarray):
            neighbor_indices = torch.from_numpy(self.ID).long().to(self.device) 
        else:
            neighbor_indices = self.ID.long()
            
        # 3. 提取邻居点坐标 (Fancy Indexing)
        flat_indices = neighbor_indices.view(-1) 
        neighbors_flat = points_3D[:, :, flat_indices]
        neighbors = neighbors_flat.view(F, 3, N, K)
        centers = points_3D.unsqueeze(-1)
        
        # 4. 计算欧氏距离矩阵 [F_batch, N, K]
        dists = torch.norm(centers - neighbors, p=2, dim=1) 
        
        # 5. 计算 Loss (Switch between ARAP and AMAP)
        
        # === ARAP 逻辑 ===
        # 这里实现 "Time Consistency ARAP"
        dists_t = dists[:-1, :, :]      # [0 ... F_batch-2]
        dists_t1 = dists[1:, :, :]      # [1 ... F_batch-1]
        loss_arap = torch.sum((dists_t - dists_t1) ** 2)
        
        # === AMAP 逻辑 ===
        # max_dists 取当前 batch 的最大值（或需要全局最大值，取决于你的实现）
        max_dists, _ = torch.max(dists, dim=0, keepdim=True) # [1, N, K]
        loss_amap = torch.sum((dists - max_dists) ** 2)
        
        return loss_amap#切换amap/arap


    def loss_subterm_smooth(self, depth):
        # 1. 修复设备报错：使用 self.normilized_point（它已经在 GPU 上了）
        # 注意：这里要确保 depth 的维度匹配 [F, 1, N] 或 [F, N]
        if depth.dim() == 2:
            depth_expanded = depth.unsqueeze(1)
        else:
            depth_expanded = depth
            
        points_3D = self.normilized_point * depth_expanded.repeat(1, 3, 1)
        
        F, _, N = points_3D.shape
        K = self.k
        
        # 2. 获取邻居索引并转到 GPU
        neighbor_indices = torch.from_numpy(self.ID).long().to(self.device)
        
        # 3. 计算可导的平滑损失 (拉普拉斯平滑)
        # 提取邻居坐标: [F, 3, N, K]
        flat_indices = neighbor_indices.view(-1)
        neighbors_flat = points_3D[:, :, flat_indices]
        neighbors = neighbors_flat.view(F, 3, N, K)
        
        # 计算每个点邻居的中心 (Centroid): [F, 3, N]
        centroid = torch.mean(neighbors, dim=3)
        
        # 损失目标：每个 3D 点应该靠近其邻居的中心，防止点云离散化
        # 这是一个完全可导的操作
        loss_smooth = torch.sum((points_3D - centroid) ** 2)
        
        return loss_smooth

    '''
    def loss_subterm_smooth(self, depth):
        points_3D = torch.from_numpy(self.normilized_point_batched) * depth.repeat(1, 3, 1)
        points_3D_1 = points_3D.cpu().detach().numpy()
        loss_subterm_smooth_value = 0
        for frame_idx in range(self.num_frames):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.transpose(points_3D_1[frame_idx,:,:]))
            pcd.estimate_normals()
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
            tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),vertex_normals=np.asarray(mesh.vertex_normals))
            #trg_mesh = Meshes([torch.from_numpy(np.asarray(mesh.vertices))], [torch.from_numpy(np.asarray(mesh.triangles))])
            #trimesh.convex.is_convex(tri_mesh)
            #loss_subterm_smooth_value=loss_subterm_smooth_value+mesh_laplacian_smoothing(trg_mesh,  "cot")

        return loss_subterm_smooth_value
    '''


    def loss_subterm_video(self, depth, frame_indices=None):
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))
        
        loss_video_value = 0
        for i in range(len(frame_indices) - 1):
            idx = frame_indices[i]
            next_idx = frame_indices[i+1]
            loss_video_value = loss_video_value + torch.linalg.norm(depth[next_idx,:] - depth[idx,:])

        return loss_video_value

    def loss_subterm_scale(self, depth, frame_indices=None):
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        # 1. 提取当前批次的深度并处理维度
        d_batch = depth[frame_indices]
        
        # 如果是 [Batch, N]，增加一个维度变成 [Batch, 1, N]
        if d_batch.dim() == 2:
            d_batch = d_batch.unsqueeze(1)
        # 如果已经是 [Batch, 1, N]，则不需要再 unsqueeze，否则会变成 4 维导致 repeat 报错
        
        # 2. 计算 3D 点坐标
        # self.normilized_point[frame_indices] 形状为 [Batch, 3, N]
        # d_batch.repeat(1, 3, 1) 将深度复制 3 份，匹配 X, Y, Z
        points_3D = self.normilized_point[frame_indices] * d_batch.repeat(1, 3, 1)
        
        loss_scale_value = 0
        # 3. 计算尺度损失：强制 Z 轴中心靠近 10（防止点云飞走）
        for i in range(len(frame_indices)):
            # points_3D[i, 2, :] 是第 i 帧所有点的 Z 轴坐标
            loss_scale_value = loss_scale_value + torch.mean(torch.abs(points_3D[i, 2, :] - 10))

        return loss_scale_value
    
    '''
    def loss_subterm_scale(self, depth, frame_indices=None):
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        points_3D = self.normilized_point[frame_indices] * depth[frame_indices].unsqueeze(1).repeat(1, 3, 1)
        loss_scale_value = 0
        for i in range(len(frame_indices)):
            loss_scale_value=loss_scale_value+torch.linalg.norm(points_3D[i,2,:]-10)

        return loss_scale_value
    '''



    def loss_all(self,ShapeDecoder,shape_latent_code,Shape_partial_derivate,model_derivation,iteration, frame_indices=None):
        #depth_code = learning_model(shape_latent_code, self.num_points)
        depth = ShapeDecoder.forward(shape_latent_code)

        combined_loss=torch.zeros(1, 1).to(self.device)
        ## First loss function about the connection invariance and metric tensor invariance
        if loss_functions_params["weight_connection"] != 0.:
            combined_loss += loss_functions_params["weight_connection"] * self.loss_subterm_connection_and_metric_tensor(depth, Shape_partial_derivate, model_derivation, frame_indices=frame_indices) * self.omega_CC_and_MC

        if loss_functions_params["weight_inextensity"] != 0.:# and iteration>=3000:
            combined_loss += torch.tensor(loss_functions_params["weight_inextensity"]).to(self.device) * self.loss_subterm_distance_invariance(depth, frame_indices=frame_indices) * torch.tensor(self.omega_ARAP).to(self.device)

        if loss_functions_params["weight_smooth"] != 0.:
            combined_loss += loss_functions_params["weight_smooth"] * self.loss_subterm_smooth(depth)

        if loss_functions_params["loss_video"] != 0.:
            combined_loss += loss_functions_params["loss_video"] * self.loss_subterm_video(depth, frame_indices=frame_indices)

        if loss_functions_params["weight_scale_limitation"] != 0.:
            combined_loss += loss_functions_params["weight_scale_limitation"] * self.loss_subterm_scale(depth, frame_indices=frame_indices)

        return combined_loss


    def loss_all_GNC(self, ShapeDecoder, shape_latent_code, Shape_partial_derivate, iteration, network_model, Initial_shape, frame_indices=None):
        # depth_code = learning_model(shape_latent_code, self.num_points)
        if network_model == "MLP":
            depth = ShapeDecoder.forward(shape_latent_code)
        elif network_model == "DGNC":
            depth = ShapeDecoder.forward(shape_latent_code) + Initial_shape
            depth = torch.unsqueeze(depth, 1)

        combined_loss = torch.zeros(1, 1).to(self.device)
        ## First loss function about the connection invariance and metric tensor invariance
        if loss_functions_params["weight_connection"] != 0.:
            combined_loss += loss_functions_params["weight_connection"] * self.loss_subterm_connection_and_metric_tensor(
                depth, Shape_partial_derivate, [], frame_indices=frame_indices) * self.omega_CC_and_MC

        if loss_functions_params["weight_inextensity"] != 0.:  # and iteration>=3000:
            combined_loss += torch.tensor(loss_functions_params["weight_inextensity"]).to(
                self.device) * self.loss_subterm_distance_invariance(depth, frame_indices=frame_indices) * torch.tensor(self.omega_ARAP).to(self.device)

        if loss_functions_params["weight_smooth"] != 0.:
            combined_loss += loss_functions_params["weight_smooth"] * self.loss_subterm_smooth(depth)

        if loss_functions_params["loss_video"] != 0.:
            combined_loss += loss_functions_params["loss_video"] * self.loss_subterm_video(depth, frame_indices=frame_indices)

        if loss_functions_params["weight_scale_limitation"] != 0.:
            combined_loss += loss_functions_params["weight_scale_limitation"] * self.loss_subterm_scale(depth, frame_indices=frame_indices)

        return combined_loss