from collections import OrderedDict
from typing import List

import torch
from torch import nn

from model.common import FFN
from model.position_embedding import PositionEmbedding2d
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
    def __init__(self, blocks: List = None, channels: List = None, k=9, gcn='MRConv2d', act=nn.GELU(), drop_prob=0.1):
        """
        :param blocks: The number of blocks in each layer.
        :param channels: The number of channels in each layer.
        :param k: The number of neighbors.
        :param gcn: The graph convolution type. (MRConv2d, EdgeConv2d, GraphSAGE, GINConv2d)
        :param act: The activation function.
        :param drop_prob:  The probability of DropPath.
        """
        super(PyramidViG, self).__init__()
        max_size = (800, 800)  # the maximum size of the input image
        min_size = (224, 224)  # the minimum size of the input image
        assert len(blocks) == len(channels)  # the length of blocks and channels must be equal
        if channels is None:
            channels = [48, 96, 240, 384]
        if blocks is None:
            blocks = [2, 2, 6, 2]
        self.blocks = blocks
        n_blocks = sum(blocks)

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, n_blocks)]  # stochastic depth decay rule
        n_knn = [int(x.item()) for x in torch.linspace(k, k, n_blocks)]  # number of neighbors
        max_dilation = (min_size[0] // 2**(1+len(blocks))) * (min_size[1] // 2**(1+len(blocks))) // max(n_knn)

        self.stem = Stem4(3, channels[0], act)
        self.pos_embed = PositionEmbedding2d()
        self.backbone = nn.Sequential()
        idx = 0
        for i in range(len(blocks)):
            block = nn.Sequential()
            if i > 0:  # DownSample
                block.append(
                    nn.Sequential(
                        nn.Conv2d(channels[i - 1], channels[i], 3, stride=2, padding=1),
                        nn.BatchNorm2d(channels[i])
                    )
                )

            for j in range(blocks[i]):
                block.append(
                    nn.Sequential(
                        Grapher(channels[i], n_knn[idx], min(idx//4+1, max_dilation), gcn, act, drop_probs[idx]),
                        FFN(channels[i], channels[i]*4, channels[i], act, drop_probs[idx])
                    )
                )
                idx += 1
            self.backbone.append(block)

    def forward(self, x, return_intermediate=False):
        """
        :param x: The input images.
        :param return_intermediate: If True, return the intermediate features. ( excluding the first layer )
                                    Otherwise, return the last features.
        :return: A single Tensor or an OrderedDict[Tensor]
        """
        x = self.stem(x)  # (batch_size, channels[0], height/4, width/4)
        x = self.pos_embed(x) + x

        features = OrderedDict()  # store the intermediate features
        for i in range(len(self.blocks)):
            x = self.backbone[i](x)  # (batch_size, channels[i], height/(4*2^i), width/(4*2^i))
            if i > 0:  # do not store features of the first layer
                features[str(i)] = x

        if return_intermediate:
            return features
        else:
            return x


class Stem16(nn.Module):
    """
    Image to Visual Word Embedding Overlap
    https://github.com/whai362/PVT
    """
    def __init__(self, in_ch=3, out_ch=768, act=nn.GELU()):
        """
        The image resolution will be divided by 16×16.
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param act: The activation function.
        """
        super(Stem16, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch // 8),
            act,
            nn.Conv2d(out_ch // 8, out_ch // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch // 4),
            act,
            nn.Conv2d(out_ch // 4, out_ch // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch // 2),
            act,
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.stem(x)  # (batch_size, out_ch, height/16, width/16)
        return x


class Stem4(nn.Module):
    """
    Image to Visual Word Embedding Overlap
    https://github.com/whai362/PVT
    """
    def __init__(self, in_ch=3, out_ch=768, act=nn.GELU()):
        """
        The image resolution will be divided by 4×4.
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param act: The activation function.
        """
        super(Stem4, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch//2),
            act,
            nn.Conv2d(out_ch//2, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            act,
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.stem(x)
        return x
