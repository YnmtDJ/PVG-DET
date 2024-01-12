import math

import torch
from torch import nn
from torch.nn import functional as F


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1):
        """
        :param k: The number of neighbors
        :param dilation: The dilation rate
        """
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k

    def forward(self, x, relative_pos):
        x = F.normalize(x, p=2.0, dim=1)  # (batch_size, num_dims, num_points, 1)
        edge_index = dense_knn_matrix(x, self.k*self.dilation, relative_pos)  # (2, batch_size, num_points, k*dilation)
        edge_index = edge_index[:, :, :, ::self.dilation]  # (2, batch_size, num_points, k)
        return edge_index


@torch.no_grad()
def dense_knn_matrix(x, k=16, relative_pos=None):
    """
    Get KNN based on the pairwise distance.
    :param x: (batch_size, num_dims, num_points, 1)
    :param k: The number of neighbors
    :param relative_pos: The relative position embedding (batch_size, num_points, num_points)
    :return: The indices of nearest neighbors and center - (2, batch_size, num_points, k)
    """
    x = x.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = x.shape
    # memory efficient implementation
    n_part = 10000
    if n_points > n_part:
        nn_idx_list = []
        groups = math.ceil(n_points / n_part)
        for i in range(groups):
            start_idx = n_part * i
            end_idx = min(n_points, n_part * (i + 1))
            x_part = x[:, start_idx:end_idx, :]  # (batch_size, n_part, num_dims)
            dist = torch.cdist(x_part, x, p=2)  # (batch_size, n_part, num_points)
            _, nn_idx_part = torch.topk(-dist, k=k)  # (batch_size, n_part, k)
            nn_idx_list += [nn_idx_part]
        nn_idx = torch.cat(nn_idx_list, dim=1)  # (batch_size, num_points, k)
    else:
        dist = torch.cdist(x, x, p=2)  # (batch_size, num_points, num_points)
        _, nn_idx = torch.topk(-dist, k=k)  # (batch_size, num_points, k)
    center_idx = torch.arange(0, n_points, device=x.device).expand(batch_size, k, -1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)  # (2, batch_size, num_points, k)


@torch.no_grad()
def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    :param x: (batch_size, num_points, num_dims)
    :return: pairwise distance - (batch_size, num_points, num_points)
    """
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))  # (batch_size, num_points, num_points)
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)  # (batch_size, num_points, 1)
    return x_square + x_inner + x_square.transpose(2, 1)

