import torch
from torch import autograd
import numpy as np

# Import Python spline fitting module
from NRSfM_core.spline_fitting import fit_python

# For backward compatibility, allow optional MATLAB engine
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None


class ChamferFunction(autograd.Function):
    @staticmethod
    def forward(ctx, depth, normilized_point_batched, m_or_smoothing, device):
        """
        Forward pass using spline fitting.
        
        Args:
            depth: (1, N) tensor of depth values
            normilized_point_batched: (3, N) numpy array of normalized points
            m_or_smoothing: Either MATLAB engine (legacy) or smoothing parameter (float)
            device: torch device
        """
        points_3D = torch.from_numpy(normilized_point_batched).to(device) * depth.repeat(3, 1)
        
        # Check if using MATLAB or Python backend
        use_matlab = MATLAB_AVAILABLE and hasattr(m_or_smoothing, 'fit_python')
        
        if use_matlab:
            # Legacy MATLAB path
            uv = matlab.double(normilized_point_batched[[0, 1], :].tolist())
            fit_result = m_or_smoothing.fit_python(uv, matlab.double(points_3D.tolist()), uv, nargout=6)
            dqu = torch.tensor(fit_result[1]).to(device)
            dqv = torch.tensor(fit_result[2]).to(device)
        else:
            # Python spline fitting path
            smoothing = m_or_smoothing if isinstance(m_or_smoothing, (int, float)) else 1e-5
            uv = normilized_point_batched[[0, 1], :]
            points_3D_np = points_3D.detach().cpu().numpy()
            
            # Fit spline and get derivatives
            _, dqu_np, dqv_np, _, _, _ = fit_python(uv, points_3D_np, uv, smoothing=smoothing)
            dqu = torch.tensor(dqu_np, dtype=torch.float32).to(device)
            dqv = torch.tensor(dqv_np, dtype=torch.float32).to(device)
        
        y1 = -dqu[2, :] / points_3D[2, :]
        y1 = y1.clone().detach().requires_grad_(True)
        y2 = -dqv[2, :] / points_3D[2, :]
        y2 = y2.clone().detach().requires_grad_(True)

        ctx.save_for_backward(depth)
        ctx.normilized_point_batched = normilized_point_batched
        ctx.m_or_smoothing = m_or_smoothing
        ctx.device = device
        ctx.y1 = y1
        ctx.y2 = y2
        return torch.cat((y1, y2), 0)

    @staticmethod
    def backward(ctx, grad_out):
        depth = ctx.saved_tensors[0]
        grad_out1 = grad_out[:depth.shape[1]]
        grad_out2 = grad_out[depth.shape[1]:]
        normilized_point_batched = ctx.normilized_point_batched
        m_or_smoothing = ctx.m_or_smoothing
        device = ctx.device
        gx = torch.zeros(1, depth.shape[1], device=device)
        
        # Finite difference for gradient computation
        for i in range(depth.shape[1]):
            dH = torch.zeros(1, depth.shape[1], device=device)
            dH[0, i] = 0.01
            f1 = ChamferFunction.apply(depth + dH, normilized_point_batched, m_or_smoothing, device)
            dy1 = (f1[:depth.shape[1]] - ctx.y1) / dH[0, i]
            dy2 = (f1[depth.shape[1]:] - ctx.y2) / dH[0, i]
            grad_y1_to_depth = torch.reshape(dy1, (torch.numel(dy1), 1))
            grad_y2_to_depth = torch.reshape(dy2, (torch.numel(dy2), 1))
            gx[0, i] = torch.matmul(grad_out1, grad_y1_to_depth) + torch.matmul(grad_out2, grad_y2_to_depth)

        return gx, None, None, None

