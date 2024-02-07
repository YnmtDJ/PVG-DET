import math

import torch
import torch.nn.functional as F
from torch import nn

from model.common import BasicConv, DropPath
from model.pos_embed import PositionEmbedding2d
from model.vig.gcn.edge import DenseDilatedKnnGraph


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_ch, k=9, dilation=1, sr_ratio=1, gcn='MRConv2d', drop_prob=0.1):
        """
        :param in_ch: The number of input channels.
        :param k: The number of neighbors.
        :param dilation: The dilation rate.
        :param sr_ratio: The spatial reduction ratio. If sr_ratio=1 or none, it means no reduction.
        :param gcn: The graph convolution type. (MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)
        :param drop_prob: DropPath probability.
        """
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0),
            nn.GroupNorm(32, in_ch),
        )
        self.graph_conv = DyGraphConv2d(in_ch, in_ch*2, k, dilation, sr_ratio, gcn)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 1, stride=1, padding=0),
            nn.GroupNorm(32, in_ch),
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)  # (batch_size, num_dims, height, width)
        x = self.graph_conv(x)  # (batch_size, 2*num_dims, height, width)
        x = self.fc2(x)  # (batch_size, num_dims, height, width)
        x = self.drop_path(x) + shortcut
        return x


class DyGraphConv2d(nn.Module):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_ch, out_ch, k=9, dilation=1, sr_ratio=1, gcn="MRConv2d"):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param k: The number of neighbors.
        :param dilation: The dilation rate.
        :param sr_ratio: The spatial reduction ratio. If sr_ratio=1 or none, it means no reduction.
        :param gcn: Graph convolution type. (MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)
        """
        super(DyGraphConv2d, self).__init__()
        self.out_ch = out_ch
        self.sr_ratio = sr_ratio
        if gcn == 'MRConv2d':
            self.gcn = MRConv2d(in_ch, out_ch)
        elif gcn == 'EdgeConv2d':
            self.gcn = EdgeConv2d(in_ch, out_ch)
        elif gcn == 'GraphSAGE':
            self.gcn = GraphSAGE(in_ch, out_ch)
        elif gcn == 'GINConv2d':
            self.gcn = GINConv2d(in_ch, out_ch)
        else:
            self.gcn = None
            raise NotImplementedError('gcn:{} is not supported'.format(gcn))
        self.dilated_knn_graph = DenseDilatedKnnGraph(k, dilation)
        self.pos_embed = PositionEmbedding2d()

    def forward(self, x):
        batch_size, in_ch, height, width = x.shape
        sr_ratio = self.sr_ratio

        # TODO: Spatial Reduction or Linear Spatial Reduction
        # get the spatial reduction of the input feature and position embedding
        pos_x = self.pos_embed(x)  # (1, d_model, height, width)
        if sr_ratio is not None:  # and sr_ratio > 1:
            # x_reduce = F.adaptive_avg_pool2d(x, (height // sr_ratio, width // sr_ratio))
            # pos_x_reduce = F.adaptive_avg_pool2d(pos_x, (height // sr_ratio, width // sr_ratio))
            alpha = math.sqrt(height / width)
            x_reduce = F.adaptive_avg_pool2d(x, (round(7 * alpha), round(7 / alpha)))
            pos_x_reduce = F.adaptive_avg_pool2d(pos_x, (round(7 * alpha), round(7 / alpha)))
        else:
            x_reduce = x
            pos_x_reduce = pos_x

        # calculate relative position and reshape the feature tensor
        relative_pos = get_2d_relative_pos(pos_x.squeeze(0), pos_x_reduce.squeeze(0))
        relative_pos = relative_pos.unsqueeze(0)  # (1, n_points, m_points)
        x = x.reshape(batch_size, in_ch, -1, 1)  # (batch_size, in_ch, n_points, 1)
        x_reduce = x_reduce.reshape(batch_size, in_ch, -1, 1)  # (batch_size, in_ch, m_points, 1)

        # dilated knn graph convolution
        edge_index = self.dilated_knn_graph(x, x_reduce, relative_pos)  # (2, batch_size, n_points, k)
        x = self.gcn(x, x_reduce, edge_index)  # (batch_size, in_ch, n_points, 1)
        return x.reshape(batch_size, -1, height, width)


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution.
    https://arxiv.org/abs/1904.03751
    """
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        """
        super(MRConv2d, self).__init__()
        self.conv = BasicConv(in_ch*2, out_ch)

    def forward(self, x, y, edge_index):
        """
        :param x: The feature tensors - (batch_size, num_dims, n_points, 1)
        :param y: THe neighbors of x which need to be selected - (batch_size, num_dims, m_points, 1)
        :param edge_index: (2, batch_size, n_points, k)
        """
        x_i = batched_index_select(x, edge_index[1])  # center (batch_size, num_dims, n_points, k)
        x_j = batched_index_select(y, edge_index[0])  # neighbors (batch_size, num_dims, n_points, k)
        x_j, _ = torch.max(x_j - x_i, dim=-1, keepdim=True)  # (batch_size, num_dims, n_points, 1)
        x = torch.cat([x, x_j], dim=1)  # (batch_size, num_dims*2, n_points, 1)
        return self.conv(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, group normalization).
    """
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        """
        super(EdgeConv2d, self).__init__()
        self.conv = BasicConv(in_ch*2, out_ch)

    def forward(self, x, y, edge_index):
        """
        :param x: The feature tensors - (batch_size, num_dims, n_points, 1)
        :param y: THe neighbors of x which need to be selected - (batch_size, num_dims, m_points, 1)
        :param edge_index: (2, batch_size, n_points, k)
        """
        x_i = batched_index_select(x, edge_index[1])  # center (batch_size, num_dims, n_points, k)
        x_j = batched_index_select(y, edge_index[0])  # neighbors (batch_size, num_dims, n_points, k)
        output, _ = torch.max(self.conv(torch.cat([x_i, x_j - x_i], dim=1)), dim=-1, keepdim=True)
        return output


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution.
    https://arxiv.org/abs/1706.02216
    """
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        """
        super(GraphSAGE, self).__init__()
        self.conv1 = BasicConv(in_ch, in_ch)
        self.conv2 = BasicConv(in_ch*2, out_ch)

    def forward(self, x, y, edge_index):
        """
        :param x: The feature tensors - (batch_size, num_dims, n_points, 1)
        :param y: THe neighbors of x which need to be selected - (batch_size, num_dims, m_points, 1)
        :param edge_index: (2, batch_size, n_points, k)
        """
        x_j = batched_index_select(y, edge_index[0])  # neighbors (batch_size, num_dims, n_points, k)
        x_j, _ = torch.max(self.conv1(x_j), dim=-1, keepdim=True)
        return self.conv2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution.
    https://arxiv.org/abs/1810.00826
    """
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        """
        super(GINConv2d, self).__init__()
        self.conv = BasicConv(in_ch, out_ch)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, y, edge_index):
        """
        :param x: The feature tensors - (batch_size, num_dims, n_points, 1)
        :param y: THe neighbors of x which need to be selected - (batch_size, num_dims, m_points, 1)
        :param edge_index: (2, batch_size, n_points, k)
        """
        x_j = batched_index_select(y, edge_index[0])  # neighbors (batch_size, num_dims, n_points, k)
        x_j = torch.sum(x_j, dim=-1, keepdim=True)
        return self.conv((1 + self.eps) * x + x_j)


def batched_index_select(x, idx):
    """
    Fetches neighbors features from a given neighbor idx
    :param x: Input feature Tensor (batch_size, num_dims, n_points, 1)
    :param idx: Edge_idx (batch_size, m_points, k)
    :return: Neighbors features (batch_size, num_dims, m_points, k)
    """

    """
    Turn the tensor into a one-dimensional, 
    so the idx in each image increases n_points from the previous image.
    x - (batch_size*n_points, num_dims)
    idx - (batch_size*m_points*k)
    feature - (batch_size*m_points*k, num_dims)
    """
    batch_size, num_dims, n_points = x.shape[:3]
    _, m_points, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).reshape(-1, 1, 1) * n_points
    idx = idx + idx_base
    idx = idx.reshape(-1)

    x = x.transpose(2, 1)
    feature = x.reshape(batch_size * n_points, -1)[idx, :]
    feature = feature.reshape(batch_size, m_points, k, num_dims).permute(0, 3, 1, 2)
    return feature


def get_2d_relative_pos(pos_x, pos_y):
    """
    Calculate the relative position between two 2d position embeddings.
    :param pos_x: (d_model, height_x, width_x)
    :param pos_y: (d_model, height_y, width_y)
    :return: relative position (height_x*width_x, height_y*width_y)
    """
    d_model, _, _ = pos_x.shape
    pos_x = pos_x.reshape(d_model, -1)  # (d_model, num_points_x)
    pos_y = pos_y.reshape(d_model, -1)  # (d_model, num_points_y)

    # calculate relative position
    # (num_points_x, num_points_y)
    relative_pos = 2 * torch.matmul(pos_x.transpose(0, 1), pos_y) / d_model
    return relative_pos
