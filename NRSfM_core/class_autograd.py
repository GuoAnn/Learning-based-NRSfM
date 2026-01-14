import torch
from torch import autograd
import numpy as np
from NRSfM_core.spline_fitting import fit_python

class ChamferFunction(autograd.Function):
    @staticmethod
    def forward(ctx, depth, normilized_point_batched, m, device):
        points_3D = torch.from_numpy(normilized_point_batched).to(device) * depth.repeat(3, 1)
        #normilized_point_batched = points_3D[[0, 1], :] / points_3D[2, :]
        uv = normilized_point_batched[[0, 1], :]
        #torch.save(points_3D.clone(), 'file.pt')
        
        # Use Python spline fitting (m is now optional/ignored if None)
        points_3D_np = points_3D.cpu().detach().numpy() if isinstance(points_3D, torch.Tensor) else points_3D
        fit_result = fit_python(uv, points_3D_np, uv, smoothing=1e-5, grid_size=50)
        
        dqu = torch.tensor(fit_result[1], dtype=torch.float32).to(device)
        dqv = torch.tensor(fit_result[2], dtype=torch.float32).to(device)
        y1 = -dqu[2, :] / points_3D[2, :]
        y1.clone().detach().requires_grad_(True)
        y2 = -dqv[2, :] / points_3D[2, :]
        y2.clone().detach().requires_grad_(True)

        ctx.save_for_backward(depth)
        ctx.normilized_point_batched = normilized_point_batched
        ctx.m = m
        ctx.y1 = y1
        ctx.y2 = y2
        ctx.device = device
        return torch.cat((y1, y2), 0)

    @staticmethod
    def backward(ctx, grad_out):
        depth = ctx.saved_tensors[0]
        grad_out1 = grad_out[:depth.shape[1]]
        grad_out2 = grad_out[depth.shape[1]:]
        normilized_point_batched = ctx.normilized_point_batched
        m = ctx.m
        device = ctx.device
        gx = torch.zeros(1, depth.shape[1])
        gy = torch.zeros(1, depth.shape[1])
        for i in range(depth.shape[1]):
            dH = torch.zeros(1, depth.shape[1])
            dH[0,i] = 0.01
            f1 = ChamferFunction.apply(depth + dH, normilized_point_batched, m, device)
            dy1 =(f1[:depth.shape[1]] - ctx.y1) / dH[0, i]
            dy2 =(f1[depth.shape[1]:] - ctx.y2) / dH[0, i]
            grad_y1_to_depth = torch.reshape(dy1, (torch.numel(dy1),1))
            grad_y2_to_depth = torch.reshape(dy2, (torch.numel(dy2),1))
            gx[0, i] = torch.matmul(grad_out1, grad_y1_to_depth)+torch.matmul(grad_out2, grad_y2_to_depth)

        return gx, None, None, None

