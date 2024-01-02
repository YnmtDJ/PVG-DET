import torch
from torch import nn

from model.common import FFN
from model.vig.gcn.grapher import Grapher


class ViG(nn.Module):
    """
    Vision GNN: An Image is Worth Graph of Nodes. (Without pyramid)
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
    """
    def __init__(self, d_model=192, act=nn.GELU(), drop_prob=0.1, n_blocks=12, k=9, use_dilation=True, gcn='MRConv2d'):
        """
        :param d_model: The number of hidden channels.
        :param act: The activation function.
        :param drop_prob: The probability of DropPath.
        :param n_blocks: The number of blocks.
        :param k: The number of neighbors.
        :param use_dilation: If True, use dilation.
        :param gcn: The graph convolution type. (MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)
        """
        super(ViG, self).__init__()
        max_size = (800, 800)  # the maximum size of the input image
        min_size = (224, 224)  # the minimum size of the input image
        self.n_blocks = n_blocks

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, n_blocks)]  # stochastic depth decay rule
        n_knn = [int(x.item()) for x in torch.linspace(k, 2*k, n_blocks)]  # number of neighbors
        max_dilation = (min_size[0]//16)*(min_size[1]//16) // max(n_knn)

        if use_dilation:
            self.backbone = nn.Sequential(*[
                nn.Sequential(
                    Grapher(d_model, n_knn[i], min(i//4+1, max_dilation), gcn, act, drop_probs[i]),
                    FFN(d_model, d_model*4, d_model, act, drop_probs[i])
                )
                for i in range(n_blocks)
            ])
        else:
            self.backbone = nn.Sequential(*[
                nn.Sequential(
                    Grapher(d_model, n_knn[i], 1, gcn, act, drop_probs[i]),
                    FFN(d_model, d_model*4, d_model, act, drop_probs[i])
                )
                for i in range(n_blocks)
            ])

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.backbone[i](x)  # (batch_size, d_model, height, width)
        return x


class PyramidViG(nn.Module):
    """
    Vision GNN: An Image is Worth Graph of Nodes. (With pyramid)
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
    """
    def __init__(self, d_model=192, act=nn.GELU(), drop_prob=0.1, blocks=None, channels=None, k=9, use_dilation=True, gcn='MRConv2d'):
        """
        TODO:
        :param d_model:
        :param act:
        :param drop_prob:
        :param blocks:
        :param channels:
        :param k:
        :param use_dilation:
        :param gcn:
        """
        super(PyramidViG, self).__init__()
        max_size = (800, 800)  # the maximum size of the input image
        min_size = (224, 224)  # the minimum size of the input image
        if channels is None:
            channels = [48, 96, 240, 384]
        if blocks is None:
            blocks = [2, 2, 6, 2]
        n_blocks = sum(blocks)

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, n_blocks)]  # stochastic depth decay rule
        n_knn = [int(x.item()) for x in torch.linspace(k, k, n_blocks)]  # number of neighbors
        max_dilation = (min_size[0] // 16) * (min_size[1] // 16) // max(n_knn)

        if use_dilation:
            self.backbone = nn.Sequential(*[
                nn.Sequential(
                    Grapher(d_model, n_knn[i], min(i // 4 + 1, max_dilation), gcn, act, drop_probs[i]),
                    FFN(d_model, d_model * 4, d_model, act, drop_probs[i])
                )
                for i in range(n_blocks)
            ])
        else:
            self.backbone = nn.Sequential(*[
                nn.Sequential(
                    Grapher(d_model, n_knn[i], 1, gcn, act, drop_probs[i]),
                    FFN(d_model, d_model * 4, d_model, act, drop_probs[i])
                )
                for i in range(n_blocks)
            ])
