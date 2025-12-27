import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.contiguous()
    x = x.view(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    idx_base = idx_base.cuda(torch.get_device(x))
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
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class ShapeDecoder(nn.Module):
    def __init__(self,num_frame,num_points,Initial_shape,device):
        super(ShapeDecoder,self).__init__()
        self.num_frame = num_frame
        self.num_points = num_points
        self.activation = nn.ELU(alpha=1.0) #nn.PReLU()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 16)
        self.fc6 = nn.Linear(16, 32)
        self.fc7 = nn.Linear(32, 32)
        self.fc8 = nn.Linear(32, 32)
        self.fc9 = nn.Linear(32, num_points, bias=False)
        self.Initial_shape = Initial_shape
        self.device=device

    def forward(self,x):
        fc1_out = self.activation(self.fc1(x))
        fc2_out = self.activation(self.fc2(fc1_out))
        fc3_out = self.activation(self.fc3(fc2_out))
        fc4_out = self.activation(self.fc4(fc3_out))
        fc5_out = self.activation(self.fc5(fc4_out))
        fc6_out = self.activation(self.fc6(fc5_out))
        fc7_out = self.activation(self.fc7(fc6_out))
        fc8_out = self.activation(self.fc8(fc7_out))
        depth = self.fc9(fc8_out) +torch.tensor((self.Initial_shape),dtype=torch.float32, device=self.device)#
        #depth = torch.tensor((self.Initial_shape))
        reconstruction = depth.reshape(self.num_frame,  1, self.num_points)
        return reconstruction


class ShapeDecoder_DGNC(nn.Module):
    def __init__(self, num_control_points, num_points=40, mode=0):
        """
        Control points prediction network. Takes points as input
        and outputs control points grid.
        :param num_control_points: size of the control points grid.
        :param num_points: number of nearest neighbors used in DGCNN.
        :param mode: different modes are used that decides different number of layers.
        """
        super(ShapeDecoder_DGNC, self).__init__()
        self.k = num_points
        self.mode = mode
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
            self.conv7 = torch.nn.Conv1d(1024, 1024, 1)

            # Predicts the entire control points grid.
            self.conv8 = torch.nn.Conv1d(1024,  self.controlpoints, 1)

            self.bn6 = nn.BatchNorm1d(1024)
            self.bn7 = nn.BatchNorm1d(1024)

        self.tanh = nn.Tanh()


    def forward(self, x, weights=None):
        """
        :param weights: weights of size B x N
        """
        batch_size = x.size(0)
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

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        if isinstance(weights, torch.Tensor):
            weights = weights.reshape((1, 1, -1))
            x = x * weights

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        x1 = torch.unsqueeze(x1, 2)

        x = F.dropout(F.relu(self.bn6(self.conv6(x1))), self.drop)

        x = F.dropout(F.relu(self.bn7(self.conv7(x))), self.drop)
        x = self.conv8(x)
        x = self.tanh(x[:, :, 0])

        #x = x.view(batch_size, self.controlpoints * self.controlpoints, 3)
        return x
