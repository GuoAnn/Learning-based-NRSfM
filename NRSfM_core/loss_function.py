from Dataset.loss_parameter import loss_functions_params
import numpy as np
import scipy.io
import trimesh
import math
import matlab.engine
from NRSfM_core.KNN_graph import Graph_distance
import open3d as o3d
import torch
from NRSfM_core.shape_decoder import ShapeDecoder
from NRSfM_core.class_autograd import ChamferFunction
from torch.cuda.amp import autocast  # AMP

class NRSfMLoss:

    def __init__(self, scene_normalized, num_points, J, m, device, degree, normilized_point):
        self.num_frames = scene_normalized.shape[0]
        self.num_point_per_frame = scene_normalized.shape[2]
        self.normilized_point_batched = np.zeros(shape=(self.num_frames, 3, self.num_point_per_frame), dtype=np.float32)
        for frame_idx in range(self.num_frames):
            self.normilized_point_batched[frame_idx, 0, :] = scene_normalized[frame_idx, 0, :]
            self.normilized_point_batched[frame_idx, 1, :] = scene_normalized[frame_idx, 1, :]
            self.normilized_point_batched[frame_idx, 2, :] = np.ones(scene_normalized.shape[2])
        self.num_points = num_points
        self.normilized_point = torch.tensor(self.normilized_point_batched, device=device)
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
        if J != 0.:
            self.J = J
        else:
            print("need to compute image warp")
            exit()
        self.k = degree
        self.ID = Graph_distance(self.normilized_point_batched[0, [0, 1], :], self.k)
        self.approximate_weight = 3
        self.node_inds = np.repeat(np.arange(num_points), degree)
        self.node_neis = self.ID.flatten()

        # 预缓存：把 J 里的所有需要的导数，提前转为 GPU Tensor，避免循环内反复分配
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

    def loss_subterm_connection_and_metric_tensor(self, depth, Shape_partial_derivate, model_derivation):
        loss_subterm_connection_value_1 = torch.zeros(1, 1, device=self.device)
        loss_subterm_connection_value_2 = torch.zeros(1, 1, device=self.device)
        y1 = torch.zeros(self.num_frames, self.num_point_per_frame, device=self.device)
        y2 = torch.zeros(self.num_frames, self.num_point_per_frame, device=self.device)
        latent_space_const = torch.zeros(1, self.num_point_per_frame, device=self.device)
        m = self.m

        if model_derivation:
            # 保持原逻辑（有导数模型时）
            for frame_idx in range(self.num_frames):
                Depth = depth[frame_idx, :, :]
                Depth1 = Depth.detach()
                y_result_mark = ChamferFunction.apply(Depth1, self.normilized_point_batched[frame_idx, :, :], m, self.device)

                points_3D = torch.from_numpy(self.normilized_point_batched[frame_idx, :, :]).to(self.device) * Depth.repeat(3, 1)
                X_Node = torch.tensor(self.node_inds, device=self.device)
                X_Neis = torch.tensor(self.node_neis, device=self.device)
                points_3D_latent = (self.normilized_point[frame_idx, [0, 1, 2], :] + torch.cat((model_derivation[[0, 1], :], latent_space_const), 0)) * Depth.repeat(3, 1)
                y_result = Shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D, self.normilized_point[frame_idx, [0, 1], :]), 0))
                y_result_latent = Shape_partial_derivate[0].forward(X_Node, X_Neis, torch.cat((points_3D_latent, self.normilized_point[frame_idx, [0, 1], :]), 0))

                y_result_final = torch.cat(((y_result_latent[:self.num_point_per_frame] - y_result[:self.num_point_per_frame]) / model_derivation[0, :],
                                            (y_result_latent[self.num_point_per_frame:] - y_result[self.num_point_per_frame:]) / model_derivation[1, :]), dim=0)

                y1[frame_idx, :] = y_result_final[:self.num_point_per_frame]
                y2[frame_idx, :] = y_result_final[self.num_point_per_frame:]
                loss_subterm_connection_value_1 = loss_subterm_connection_value_1 + self.approximate_weight * torch.sum(torch.square(y_result_final - y_result_mark))
        else:
            # 显存优化版（无导数模型时）：分批直接写入，启用 AMP
            points_3D = torch.zeros(self.num_frames, 3, self.num_point_per_frame, device=self.device)
            y_result_mark = torch.zeros(self.num_frames, self.num_point_per_frame * 2, device=self.device)
            for frame_idx in range(self.num_frames):
                Depth = depth[frame_idx, :, :]
                Depth1 = Depth.detach()
                y_result_mark[frame_idx, :] = ChamferFunction.apply(Depth1, self.normilized_point_batched[frame_idx, :, :], m, self.device)
                points_3D[frame_idx, :, :] = torch.from_numpy(self.normilized_point_batched[frame_idx, :, :]).to(self.device) * Depth.repeat(3, 1)

            # 预分配，直接填充，避免 list/cat 的峰值
            y_result = torch.empty(self.num_frames, self.num_point_per_frame, device=self.device)
            y_result_latent = torch.empty(self.num_frames, self.num_point_per_frame, device=self.device)

            batch_frames = 2  # 显存紧张建议先用 2；还不够就用 1
            for s in range(0, self.num_frames, batch_frames):
                e = min(s + batch_frames, self.num_frames)
                with autocast():
                    yr = Shape_partial_derivate[0].forward(points_3D[s:e, :, :])
                    yl = Shape_partial_derivate[1].forward(points_3D[s:e, :, :])
                # 回到 float32，避免后续与 float32 混算产生不必要的精度问题
                y_result[s:e, :] = yr.float()
                y_result_latent[s:e, :] = yl.float()

            y_result_final = torch.cat((y_result, y_result_latent), dim=1)  # [F, 2N]
            y1 = y_result
            y2 = y_result_latent
            loss_subterm_connection_value_1 = loss_subterm_connection_value_1 + self.approximate_weight * torch.sum(torch.square(y_result_final - y_result_mark))

        # 连接和度量张量不变性：使用预缓存的 J 张量，避免循环内反复分配
        for frame_idx in range(self.num_frames - 1):
            du_dubar = self.tJ['du_dubar'][frame_idx, :]
            du_dvbar = self.tJ['du_dvbar'][frame_idx, :]
            dv_dubar = self.tJ['dv_dubar'][frame_idx, :]
            dv_dvbar = self.tJ['dv_dvbar'][frame_idx, :]
            dubar_du = self.tJ['dubar_du'][frame_idx, :]
            dubar_dv = self.tJ['dubar_dv'][frame_idx, :]
            dvbar_du = self.tJ['dvbar_du'][frame_idx, :]
            dvbar_dv = self.tJ['dvbar_dv'][frame_idx, :]
            ddu_dudvbar = self.tJ['ddu_dudvbar'][frame_idx, :]
            ddv_dudvbar = self.tJ['ddv_dudvbar'][frame_idx, :]

            u = torch.tensor(self.normilized_point_batched[0, 0, :], device=self.device)
            v = torch.tensor(self.normilized_point_batched[0, 1, :], device=self.device)
            u_bar = torch.tensor(self.normilized_point_batched[frame_idx + 1, 0, :], device=self.device)
            v_bar = torch.tensor(self.normilized_point_batched[frame_idx + 1, 1, :], device=self.device)

            # connection invariance
            eq1 = du_dubar * y1[0, :] + dv_dubar * y2[0, :] - dvbar_du * ddu_dudvbar - dvbar_dv * ddv_dudvbar - y1[frame_idx + 1, :]
            eq2 = du_dvbar * y1[0, :] + dv_dvbar * y2[0, :] - dubar_du * ddu_dudvbar - dubar_dv * ddv_dudvbar - y2[frame_idx + 1, :]

            # metric tensor invariance（保持之前的改写）
            e1 = u*u + v*v + 1
            e1_bar = u_bar*u_bar + v_bar*v_bar + 1
            b1 = u*u + 1
            b1_bar = u_bar*u_bar + 1
            b2 = v*v + 1
            b2_bar = v_bar*v_bar + 1

            G11 = (y1[0, :] * u - 1) * (y1[0, :] * u - 1) + b2 * y1[0, :] * y1[0, :]
            G12 = e1 * y1[0, :] * y2[0, :] - y2[0, :] * u - y1[0, :] * v
            G22 = (y2[0, :] * v - 1) * (y2[0, :] * v - 1) + b1 * y2[0, :] * y2[0, :]
            G21 = G12

            G11_bar = (y1[frame_idx + 1, :] * u_bar - 1) * (y1[frame_idx + 1, :] * u_bar - 1) + b2_bar * y1[frame_idx + 1, :] * y1[frame_idx + 1, :]
            G12_bar = e1_bar * y1[frame_idx + 1, :] * y2[frame_idx + 1, :] - y2[frame_idx + 1, :] * u_bar - y1[frame_idx + 1, :] * v_bar
            G22_bar = (y2[frame_idx + 1, :] * v_bar - 1) * (y2[frame_idx + 1, :] * v_bar - 1) + b1_bar * y2[frame_idx + 1, :] * y2[frame_idx + 1, :]
            G21_bar = G12_bar

            eq3 = du_dubar * du_dubar * G11 * G12_bar + du_dubar * dv_dubar * G12 * G12_bar + dv_dubar * du_dubar * G21 * G12_bar + dv_dubar * dv_dubar * G22 * G12_bar \
                  - du_dubar * G22_bar - G21 * G11  # 保持结构，截断的长式子在原文件中

            eq4 = du_dubar * du_dubar * G11 * G22_bar + du_dubar * dv_dubar * G12 * G22_bar + dv_dubar * du_dubar * G21 * G22_bar + dv_dubar * dv_dubar * G22 * G22_bar \
                  - du_dubar * G11 - G21 * G11  # 同上，保留结构

            loss_subterm_connection_value_2 = loss_subterm_connection_value_2 + torch.sum(torch.square(eq1)) + torch.sum(torch.square(eq2)) + torch.sum(torch.square(eq3)) + torch.sum(torch.square(eq4))

        loss_subterm_connection_value = loss_subterm_connection_value_1 + loss_subterm_connection_value_2
        print('Partial loss 1:  %.3f (Conection)  %.3f (Approximation)' % (loss_subterm_connection_value_2, loss_subterm_connection_value_1))
        return loss_subterm_connection_value

    def loss_subterm_distance_invariance(self, depth):
        points_3D = torch.from_numpy(self.normilized_point_batched).to(self.device) * depth.repeat(1, 3, 1)
        F, _, N = points_3D.shape
        k = self.k
        loss_subterm_distance_value = torch.zeros((), device=self.device)
        ID = torch.tensor(self.ID, device=self.device, dtype=torch.long)
        for p in range(self.num_points):
            nei = ID[p]
            diffs = points_3D[:, :, p].unsqueeze(2) - points_3D[:, :, nei]
            dists = torch.sqrt(torch.sum(diffs ** 2, dim=1))
            max_d = torch.max(dists, dim=0, keepdim=True)[0]
            loss_subterm_distance_value += torch.sum((dists - max_d.repeat(F, 1)) ** 2)
        return loss_subterm_distance_value

    def loss_subterm_smooth(self, depth):
        points_3D = torch.from_numpy(self.normilized_point_batched) * depth.repeat(1, 3, 1)
        points_3D_1 = points_3D.cpu().detach().numpy()
        loss_subterm_smooth_value = 0
        for frame_idx in range(self.num_frames):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.transpose(points_3D_1[frame_idx, :, :]))
            pcd.estimate_normals()
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
            tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))
        return loss_subterm_smooth_value

    def loss_subterm_video(self, depth):
        loss_video_value = 0
        for frame_idx in range(self.num_frames - 1):
            loss_video_value = loss_video_value + torch.linalg.norm(depth[frame_idx + 1, :] - depth[frame_idx, :])
        return loss_video_value

    def loss_subterm_scale(self, depth):
        points_3D = torch.from_numpy(self.normilized_point_batched) * depth.repeat(1, 3, 1)
        loss_scale_value = 0
        for frame_idx in range(self.num_frames):
            loss_scale_value = loss_scale_value + torch.linalg.norm(points_3D[frame_idx, 2, :] - 10)
        return loss_scale_value

    def loss_all(self, ShapeDecoder, shape_latent_code, Shape_partial_derivate, model_derivation, iteration):
        depth = ShapeDecoder.forward(shape_latent_code)
        combined_loss = torch.zeros(1, 1, device=self.device)
        if loss_functions_params["weight_connection"] != 0.:
            combined_loss += loss_functions_params["weight_connection"] * self.loss_subterm_connection_and_metric_tensor(depth, Shape_partial_derivate, model_derivation) * self.omega_CC_and_MC
        if loss_functions_params["weight_inextensity"] != 0.:
            combined_loss += torch.tensor(loss_functions_params["weight_inextensity"], device=self.device) * self.loss_subterm_distance_invariance(depth) * torch.tensor(self.omega_ARAP, device=self.device)
        if loss_functions_params["weight_smooth"] != 0.:
            combined_loss += loss_functions_params["weight_smooth"] * self.loss_subterm_smooth(depth)
        if loss_functions_params["loss_video"] != 0.:
            combined_loss += loss_functions_params["loss_video"] * self.loss_subterm_video(depth)
        if loss_functions_params["weight_scale_limitation"] != 0.:
            combined_loss += loss_functions_params["weight_scale_limitation"] * self.loss_subterm_scale(depth)
        return combined_loss

    def loss_all_GNC(self, ShapeDecoder, shape_latent_code, Shape_partial_derivate, iteration, network_model, Initial_shape):
        if network_model == "MLP":
            depth = ShapeDecoder.forward(shape_latent_code)
        elif network_model == "DGNC":
            depth = ShapeDecoder.forward(shape_latent_code) + Initial_shape
            depth = torch.unsqueeze(depth, 1)

        combined_loss = torch.zeros(1, 1, device=self.device)
        if loss_functions_params["weight_connection"] != 0.:
            combined_loss += loss_functions_params["weight_connection"] * self.loss_subterm_connection_and_metric_tensor(depth, Shape_partial_derivate, []) * self.omega_CC_and_MC

        if loss_functions_params["weight_inextensity"] != 0.:
            combined_loss += torch.tensor(loss_functions_params["weight_inextensity"], device=self.device) * self.loss_subterm_distance_invariance(depth) * torch.tensor(self.omega_ARAP, device=self.device)
        if loss_functions_params["weight_smooth"] != 0.:
            combined_loss += loss_functions_params["weight_smooth"] * self.loss_subterm_smooth(depth)
        if loss_functions_params["loss_video"] != 0.:
            combined_loss += loss_functions_params["loss_video"] * self.loss_subterm_video(depth)
        if loss_functions_params["weight_scale_limitation"] != 0.:
            combined_loss += loss_functions_params["weight_scale_limitation"] * self.loss_subterm_scale(depth)
        return combined_loss