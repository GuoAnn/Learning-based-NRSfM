import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = np.finfo(np.float32).eps


'''
def knn(x, k):
    batch_size = x.shape[0]
    indices = np.arange(0, k)
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        idx = distances.topk(k=k, dim=-1)[1][:, :, indices]
    return idx'''

def knn(x, k):
    """
    Compute kNN indices per batch on CPU to avoid large N×N GPU allocations.
    x: [B, C, N]
    return: idx [B, N, k] on x.device
    """
    device = x.device
    B, C, N = x.shape
    # 转 CPU，按批处理，避免大矩阵留在 GPU
    x_cpu = x.detach().to('cpu').transpose(1, 2).contiguous()  # [B, N, C]

    idx_list = []
    with torch.no_grad():
        for b in range(B):
            # cdist 计算欧氏距离，返回 [N, N]
            dist = torch.cdist(x_cpu[b], x_cpu[b], p=2)
            # 取最小的 k 个（最近邻）
            idx_b = dist.topk(k=k, dim=-1, largest=False).indices  # [N, k]
            idx_list.append(idx_b.unsqueeze(0))  # [1, N, k]

    idx = torch.cat(idx_list, dim=0).to(device)  # [B, N, k] 回到原设备
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()
    x = x.view(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size,device=x.device).view(-1, 1, 1) * num_points
    #idx_base = idx_base.cuda(torch.get_device(x))
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
    except:
        import ipdb;
        ipdb.set_trace()
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).expand(-1,-1,k,-1) #.repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNNControlPoints(nn.Module):
    def __init__(self, num_control_points, num_points=40, mode=0):
        """
        Control points prediction network. Takes points as input
        and outputs control points grid.
        :param num_control_points: size of the control points grid.
        :param num_points: number of nearest neighbors used in DGCNN.
        :param mode: different modes are used that decides different number of layers.
        """
        super(DGCNNControlPoints, self).__init__()
        self.k = num_points
        self.mode = 3 #要改mode值改这里就可以了 0/3
        if self.mode == 0:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024*2, 1024, 1)
            #self.conv7 = torch.nn.Conv1d(1024, 1024, 1)
            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024,  self.controlpoints, 1)

            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        if self.mode == 1:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm1d(1024)
            self.drop = 0.0

            self.conv1 = nn.Sequential(nn.Conv2d(6, 128, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv2 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv3 = nn.Sequential(nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv5 = nn.Sequential(nn.Conv1d(1024 + 128 + 3, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(1024, 1024, 1)
            self.conv7 = torch.nn.Conv1d(1024*2, 1024, 1)
            #self.conv7 = torch.nn.Conv1d(1024*2, 1024, 1)
            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024, 3 * (self.controlpoints ** 2), 1)
            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        self.tanh = nn.Tanh()

        if self.mode == 3:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm1d(2048)
            self.drop = 0.0
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(963, 2048, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.controlpoints = num_control_points
            self.conv6 = torch.nn.Conv1d(2048, 2048, 1)
            self.conv7 = torch.nn.Conv1d(2048*2, 2048*2, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(2048*2,  self.controlpoints, 1)

            self.bn6 = nn.BatchNorm1d(2048)
            self.bn7 = nn.BatchNorm1d(2048*2)

    def forward(self, x, weights=None):
        """
        :param weights: weights of size B x N
        """
        batch_size = x.size(0)
        x0 = x
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        if self.mode == 0:
            x = torch.cat((x1, x2, x3, x4), dim=1)
        else:
            x = torch.cat((x0, x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if isinstance(weights, torch.Tensor):
            weights = weights.reshape((1, 1, -1))
            x = x * weights

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x1 = torch.unsqueeze(x1, 2)

        x = F.dropout(F.relu(self.bn6(self.conv6(x1))), self.drop)

        x = torch.cat((x, x1), dim=1)

        x = F.dropout(F.relu(self.bn7(self.conv7(x))), self.drop)


        x = self.conv8(x)
        x = self.tanh(x[:, :, 0])

        #x = x.view(batch_size, self.controlpoints * self.controlpoints, 3)
        return x
