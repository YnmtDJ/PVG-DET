import torch
from torch import nn
import torch.nn.functional as F

from model.common import BasicConv, DropPath
from model.pos_embed import RelativePositionEmbedding2d, PositionEmbedding2d
from model.vig.gcn.edge import DenseDilatedKnnGraph


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_ch, k=9, dilation=1, stride=1, gcn='MRConv2d', drop_prob=0.1):
        """
        :param in_ch: The number of input channels.
        :param k: The number of neighbors.
        :param dilation: The dilation rate.
        :param stride: The stride of partition. If stride=1 or none, it means no partition.
        :param gcn: The graph convolution type. (MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)
        :param drop_prob: DropPath probability.
        """
        super(Grapher, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0),
            nn.GroupNorm(32, in_ch),
        )
        self.graph_conv = DyGraphConv2d(in_ch, in_ch*2, k, dilation, stride, gcn)
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
    def __init__(self, in_ch, out_ch, k=9, dilation=1, stride=1, gcn="MRConv2d"):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param k: The number of neighbors.
        :param dilation: The dilation rate.
        :param stride: The stride of partition. If stride=1 or none, it means no partition.
        :param gcn: Graph convolution type. (MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)
        """
        super(DyGraphConv2d, self).__init__()
        self.out_ch = out_ch
        self.stride = stride
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
        self.relative_pos_embed = RelativePositionEmbedding2d()
        self.pos_embed = PositionEmbedding2d()

    def forward(self, x):
        batch_size, in_ch, height, width = x.shape
        stride = self.stride
        pos_x = self.pos_embed(x)

        # TODO: Spatial Reduction or Linear Spatial Reduction
        x_reduce = F.adaptive_avg_pool2d(x, (height // stride, width // stride))
        pos_x_reduce = F.adaptive_avg_pool2d(pos_x, (height // stride, width // stride))








        # partition-knn implementation
        if self.stride is not None and self.stride > 1:
            x_merge = torch.zeros([batch_size, self.out_ch, height, width], device=x.device)
            stride = self.stride  # the stride of partition
            for i in range(stride):
                for j in range(stride):
                    x_part = x[:, :, i::stride, j::stride]  # (batch_size, in_ch, h_part, w_part)
                    _, _, h_part, w_part = x_part.shape
                    relative_pos = self.relative_pos_embed(x_part)  # (batch_size, h_part*w_part, h_part*w_part)
                    x_part = x_part.reshape(batch_size, in_ch, -1, 1)
                    edge_index = self.dilated_knn_graph(x_part, relative_pos)  # (2, batch_size, num_points, k)
                    x_part = self.gcn(x_part, edge_index)
                    x_part = x_part.reshape(batch_size, -1, h_part, w_part)
                    x_merge[:, :, i::stride, j::stride] += x_part
            return x_merge
        else:
            relative_pos = self.relative_pos_embed(x)  # (batch_size, height*width, height*width)
            x = x.reshape(batch_size, in_ch, -1, 1)
            edge_index = self.dilated_knn_graph(x, relative_pos)  # (2, batch_size, num_points, k)
            x = self.gcn(x, edge_index)
            return x.reshape(batch_size, -1, height, width)

    def get_2d_relative_pos(self, pos_x, pos_y):
        """
        Calculate the relative position between two 2d position embeddings.
        :param pos_x: (1, d_model_x, height_x, width_x)
        :param pos_y: (1, d_model_y, height_y, width_y)
        :return: relative position (height_x*width_x, height_y*width_y)
        """
        # TODO: Implement this function


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

    def forward(self, x, edge_index):
        """
        :param x: (batch_size, num_dims, num_points, 1)
        :param edge_index: (2, batch_size, num_points, k)
        """
        x_i = batched_index_select(x, edge_index[1])  # center (batch_size, num_dims, num_points, k)
        x_j = batched_index_select(x, edge_index[0])  # neighbors (batch_size, num_dims, num_points, k)
        x_j, _ = torch.max(x_j - x_i, dim=-1, keepdim=True)  # (batch_size, num_dims, num_points, 1)
        x = torch.cat([x, x_j], dim=1)  # (batch_size, num_dims*2, num_points, 1)
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

    def forward(self, x, edge_index):
        """
        :param x: (batch_size, num_dims, num_points, 1)
        :param edge_index: (2, batch_size, num_points, k)
        """
        x_i = batched_index_select(x, edge_index[1])  # center (batch_size, num_dims, num_points, k)
        x_j = batched_index_select(x, edge_index[0])  # neighbors (batch_size, num_dims, num_points, k)
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

    def forward(self, x, edge_index):
        """
        :param x: (batch_size, num_dims, num_points, 1)
        :param edge_index: (2, batch_size, num_points, k)
        """
        x_j = batched_index_select(x, edge_index[0])  # neighbors (batch_size, num_dims, num_points, k)
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

    def forward(self, x, edge_index):
        """
        :param x: (batch_size, num_dims, num_points, 1)
        :param edge_index: (2, batch_size, num_points, k)
        """
        x_j = batched_index_select(x, edge_index[0])  # neighbors (batch_size, num_dims, num_points, k)
        x_j = torch.sum(x_j, dim=-1, keepdim=True)
        return self.conv((1 + self.eps) * x + x_j)


def batched_index_select(x, idx):
    """
    Fetches neighbors features from a given neighbor idx
    :param x: Input feature Tensor (batch_size, num_dims, num_points, 1)
    :param idx: Edge_idx (batch_size, num_points, k)
    :return: Neighbors features (batch_size, num_dims, num_points, k)
    """

    """
    Turn the tensor into a one-dimensional, 
    so the idx in each image increases num_points from the previous image.
    x - (batch_size*num_points, num_dims)
    idx - (batch_size*num_points*k)
    feature - (batch_size*num_points*k, num_dims)
    """
    batch_size, num_dims, num_points = x.shape[:3]
    _, _, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).reshape(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.reshape(-1)

    x = x.transpose(2, 1)
    feature = x.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.reshape(batch_size, num_points, k, num_dims).permute(0, 3, 1, 2)
    return feature
