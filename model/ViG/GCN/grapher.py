import torch
from torch import nn
from torch.functional import F

from model.ViG.GCN.edge import DenseDilatedKnnGraph


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_ch, out_ch, conv='mr', act=nn.GELU(), norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'mr':
            self.gcn = MRConv2d(in_ch, out_ch, act, norm, bias)
        elif conv == 'edge':
            self.gcn = EdgeConv2d(in_ch, out_ch, act, norm, bias)
        elif conv == 'sage':
            self.gcn = GraphSAGE(in_ch, out_ch, act, norm, bias)
        elif conv == 'gin':
            self.gcn = GINConv2d(in_ch, out_ch, act, norm, bias)
        else:
            raise NotImplementedError('gcn:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gcn(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x)
        x = super(DyGraphConv2d, self).forward(x, edge_index)
        return x.reshape(B, -1, H, W).contiguous()


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution.
    https://arxiv.org/abs/1904.03751
    """
    def __init__(self, in_ch, out_ch, act=nn.GELU()):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param act: The activation function.
        """
        super(MRConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 1, groups=4),
            nn.BatchNorm2d(out_ch),
            act,
        )

    def forward(self, x, edge_index):
        """
        :param x: (batch_size, num_dims, num_points, 1)
        :param edge_index: (2, batch_size, num_points, k)
        """
        x_i = batched_index_select(x, edge_index[1])  # center (batch_size, num_dims, num_points, k)
        x_j = batched_index_select(x, edge_index[0])  # neighbors (batch_size, num_dims, num_points, k)
        x_j, _ = torch.max(x_j - x_i, dim=-1, keepdim=True)  # (batch_size, num_dims, num_points, 1)
        b, c, n, _ = x.shape
        x = torch.cat([x, x_j], dim=1)
        return self.conv(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


def batched_index_select(x, idx):
    """
    Fetches neighbors features from a given neighbor idx
    :param x: Input feature Tensor (batch_size, num_dims, num_points, 1)
    :param idx: Edge_idx (batch_size, num_points, k)
    :return: Neighbors features (batch_size, num_dims, num_points, k)
    """

    """
    Turn the tensor into a one-dimensional, 
    so the idx in each batch increases batch_size from the previous batch
    x - (batch_size*num_points, num_dims)
    idx - (batch_size*num_points*k)
    feature - (batch_size*num_points*k, num_dims)
    """
    batch_size, num_dims, num_points = x.shape[:3]
    _, _, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature
