import numpy as np
#import tensorflow as tf
import scipy.io
from Dataset.dataset_setting import dataset_params
import os
import torch as to

def load_preprocessed_W(W_file_location, data_type):
    W_preprocessed = np.loadtxt(W_file_location, dtype=data_type)
    return W_preprocessed

def get_batched_W(normilized_point, device):
    num_frames = normilized_point.shape[0] // 2
    num_point_per_frame = normilized_point.shape[1]
    normilized_point_batched=np.zeros(shape=(num_frames,2,num_point_per_frame), dtype=np.float32)

    point_rows = 0
    for frame_idx in range(num_frames):
        normilized_point_batched[frame_idx, :, :] = normilized_point[[point_rows,point_rows + 1], :]
        point_rows += 2

    normilized_point_batched_tensor = to.tensor(normilized_point_batched).to(device)
    return normilized_point_batched,normilized_point_batched_tensor


def normalized_points_downsample(W, K_inv, downsample_size):
    point_all_size = np.random.permutation(range(W.shape[1]))
    W_reduced = np.array([[0.0] * downsample_size] * W.shape[0])
    Scene=np.array([[0.0] * downsample_size] * W.shape[0])
    point_all_size_reduced=point_all_size[0:downsample_size]
    point_all_size_sorted = point_all_size_reduced[np.argsort(point_all_size_reduced)]
    for i in range(downsample_size):
        W_reduced[:,i]=W[:,int(point_all_size_sorted[i])]
    for i in range(int(W.shape[0]/2)):
        Scene[2 * i, :] = W_reduced[2 * i, :] / K_inv[0]
        Scene[2 * i+1, :] = W_reduced[2 * i+1, :] / K_inv[1]
    return Scene

def normalized_points_without_downsample(W, K_inv):
    Scene=np.array([[0.0] * W.shape[1]] * W.shape[0])
    for i in range(int(W.shape[0]/2)):
        Scene[2 * i, :] = W[2 * i, :] / K_inv[0]
        Scene[2 * i+1, :] = W[2 * i+1, :] / K_inv[1]
    return Scene

class test(object):
    __slots__ = ['dx1_dy1', 'dx1_dy2', 'dx2_dy1', 'dx2_dy2','ddx1_ddy1','ddx1_ddy2',
                 'ddx2_ddy1','ddx2_ddy2','ddx1_dxdy','ddx2_dxdy','dy1_dx1','dy2_dx1','dy1_dx2','dy2_dx2']

class test1(object):
    __slots__ = ['m', 'Pgth', 'm1', 'Pgth1']

def normalized_points_downsample_load(filename):
    mat = scipy.io.loadmat(filename)
    J_mat = mat['J']
    J=test()
    J.dx1_dy1 = J_mat['dx1_dy1'][0,0]
    J.dx1_dy2 = J_mat['dx1_dy2'][0,0]
    J.dx2_dy1 = J_mat['dx2_dy1'][0,0]
    J.dx2_dy2 = J_mat['dx2_dy2'][0,0]
    J.ddx1_ddy1 = J_mat['ddx1_ddy1'][0,0]
    J.ddx1_ddy2 = J_mat['ddx1_ddy2'][0,0]
    J.ddx2_ddy1 = J_mat['ddx2_ddy1'][0,0]
    J.ddx2_ddy2 = J_mat['ddx2_ddy2'][0,0]
    J.ddx1_dxdy = J_mat['ddx1_dxdy'][0,0]
    J.ddx2_dxdy = J_mat['ddx2_dxdy'][0,0]
    J.dy1_dx1 = J_mat['dy1_dx1'][0,0]
    J.dy2_dx1 = J_mat['dy2_dx1'][0,0]
    J.dy1_dx2 = J_mat['dy1_dx2'][0,0]
    J.dy2_dx2 = J_mat['dy2_dx2'][0,0]
    scene_mat = mat['scene']
    scene=test1()

    m_mat=scene_mat['m'][0,0]
    Pgth_mat = scene_mat['Pgth'][0, 0]
    num_frames=m_mat.shape[1]
    num_points=m_mat[0,0][0].shape[1]
    Scene = np.array([[0.0] * num_points] * num_frames*2)
    Pgth = np.zeros((num_frames, 3, num_points))
    for frame_idx in range(num_frames):
        Scene[2 * frame_idx, :] = m_mat[0,frame_idx][0][0,:]
        Scene[2 * frame_idx+1, :] = m_mat[0, frame_idx][0][1,:]
        Pgth[frame_idx, 0, :] = Pgth_mat[0, frame_idx][0][0,:]
        Pgth[frame_idx, 1, :] = Pgth_mat[0, frame_idx][0][1,:]
        Pgth[frame_idx, 2, :] = Pgth_mat[0, frame_idx][0][2,:]
    #    scene.m = scene_mat['m']
    #    scene.m1 = scene_mat['m1']
    #    scene.Pgth = scene_mat['Pgth']
    #    scene.Pgth1 = scene_mat['Pgth1']

    #scene.m()
    return Scene, Pgth, J


