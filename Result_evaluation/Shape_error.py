import numpy as np
import scipy
import matlab.engine
import torch

def shape_error(Estimation_all, Groundtruth_all, m):
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        accuracy[0,i] = view_shape(Estimation, Groundtruth, m)

    return np.mean(accuracy)


def view_shape(Shape_A, Shape_B, m):
    Shape_A_matlab = matlab.double(Shape_A.tolist())
    Shape_B_matlab = matlab.double(Shape_B.tolist())
    #error_result = m.draw_image_dense(Shape_A_matlab,Shape_B_matlab)
    error_result = m.draw_image_sparse(Shape_A_matlab, Shape_B_matlab, nargout=3)
    error_np = np.array(error_result[0])
    #P2 = np.array(error_result[1])
    #scale = np.array(error_result[2])

    return error_np


def shape_error_image(Estimation_all, Groundtruth_all, m):
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        accuracy[0,i] = view_shape_image(Estimation, Groundtruth, m)

    return np.mean(accuracy)


def view_shape_image(Shape_A, Shape_B, m):
    Shape_A_matlab = matlab.double(Shape_A.tolist())
    Shape_B_matlab = matlab.double(Shape_B.tolist())
    #error_result = m.draw_image_dense(Shape_A_matlab,Shape_B_matlab)
    error_result = m.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
    error_np = np.array(error_result[0])
    P2 = np.array(error_result[1])
    scale = np.array(error_result[2])

    return error_np

def shape_error_save(Estimation_all, Groundtruth_all, m, ):
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        Shape_A_matlab = matlab.double(Groundtruth.tolist())
        Shape_B_matlab = matlab.double(Estimation.tolist())
        error_result = m.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
        accuracy[0,i] = np.array(error_result[0])
    accuracy_tensor = torch.tensor(accuracy)
    Estimation_all_tensor = torch.tensor(Estimation_all)

    return accuracy_tensor, Estimation_all_tensor



'''def shape_error(Estimation_all, Groundtruth_all, m):
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    scales = np.zeros(Estimation_all.shape[0])          # ← 新增
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]
        Estimation = Estimation_all[i,:,:]
        acc, sc = view_shape(Estimation, Groundtruth, m) # ← 改这里
        accuracy[0,i] = acc
        scales[i] = sc                                   # ← 新增

    # ========== 打印 GT scale ==========
    print(f"[GT normalization scale] per-frame: "
          f"mean={np.mean(scales):.6f}, "
          f"min={np.min(scales):.6f}, "
          f"max={np.max(scales):.6f}, "
          f"frame0={scales[0]:.6f}")
    # ====================================

    return np.mean(accuracy)


def view_shape(Shape_A, Shape_B, m):
    Shape_A_matlab = matlab.double(Shape_A.tolist())
    Shape_B_matlab = matlab.double(Shape_B.tolist())
    error_result = m.draw_image_sparse(Shape_A_matlab, Shape_B_matlab, nargout=3)
    error_np = np.array(error_result[0])
    scale = np.array(error_result[2])        # ← 取消注释
    return error_np, float(scale)            # ← 返回 scale


def shape_error_image(Estimation_all, Groundtruth_all, m):
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        accuracy[0,i] = view_shape_image(Estimation, Groundtruth, m)

    return np.mean(accuracy)


def view_shape_image(Shape_A, Shape_B, m):
    Shape_A_matlab = matlab.double(Shape_A.tolist())
    Shape_B_matlab = matlab.double(Shape_B.tolist())
    #error_result = m.draw_image_dense(Shape_A_matlab,Shape_B_matlab)
    error_result = m.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
    error_np = np.array(error_result[0])
    P2 = np.array(error_result[1])
    scale = np.array(error_result[2])

    return error_np

def shape_error_save(Estimation_all, Groundtruth_all, m, ):
    accuracy = np.zeros(shape=(1,Estimation_all.shape[0]), dtype=np.float32)
    for i in range(Estimation_all.shape[0]):
        Groundtruth = Groundtruth_all[i,:,:]#.transpose() for dense
        Estimation = Estimation_all[i,:,:]#.transpose()
        # Draw image
        Shape_A_matlab = matlab.double(Groundtruth.tolist())
        Shape_B_matlab = matlab.double(Estimation.tolist())
        error_result = m.draw_image_sparse_with_image(Shape_A_matlab, Shape_B_matlab, nargout=3)
        accuracy[0,i] = np.array(error_result[0])
    accuracy_tensor = torch.tensor(accuracy)
    Estimation_all_tensor = torch.tensor(Estimation_all)

    return accuracy_tensor, Estimation_all_tensor'''

