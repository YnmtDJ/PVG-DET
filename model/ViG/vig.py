import numbers
from typing import Union, Sequence

import torch
from timm.layers import DropPath
from torch import nn

from model.ViG.GCN.grapher import Grapher


class ViG(nn.Module):
    """
    Vision GNN: An Image is Worth Graph of Nodes. (Without pyramid)
    https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
    """
    def __init__(self, in_ch=3, out_ch=192, act=nn.GELU(), drop_prob=0.1, n_blocks=12, k=9, use_dilation=True,
                 gcn='MRConv2d'):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
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

        self.stem = Stem(in_ch, out_ch, act)
        self.pos_embed = nn.Parameter(torch.zeros(1, out_ch, max_size[0]//16, max_size[1]//16))

        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, n_blocks)]  # stochastic depth decay rule
        n_knn = [int(x.item()) for x in torch.linspace(k, 2*k, n_blocks)]  # number of neighbors
        max_dilation = (min_size[0]//16)*(min_size[1]//16) // max(n_knn)

        if use_dilation:
            self.backbone = nn.Sequential(*[
                nn.Sequential(
                    Grapher(out_ch, n_knn[i], min(i//4+1, max_dilation), gcn, act, drop_probs[i]),
                    FFN(out_ch, out_ch*4, out_ch, act, drop_probs[i])
                )
                for i in range(n_blocks)
            ])
        else:
            self.backbone = nn.Sequential(*[
                nn.Sequential(
                    Grapher(out_ch, n_knn[i], 1, gcn, act, drop_probs[i]),
                    FFN(out_ch, out_ch*4, out_ch, act, drop_probs[i])
                )
                for i in range(n_blocks)
            ])

        self.model_init()

    def forward(self, inputs):
        x = self.stem(inputs)  # (batch_size, out_ch, height/16, width/16)
        x = x + self.pos_embed[:, :, :x.shape[2], :x.shape[3]]
        for i in range(self.n_blocks):
            x = self.backbone[i](x)  # (batch_size, out_ch, height/16, width/16)
        return x

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True


class Stem(nn.Module):
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
        super(Stem, self).__init__()
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


class FFN(nn.Module):
    """
    Feed Forward Network
    """
    def __init__(self, in_features, hidden_features, out_features, act=nn.GELU(), drop_prob=0.1):
        """
        :param in_features: The number of input channels.
        :param hidden_features: The number of hidden channels.
        :param out_features: The number of output channels.
        :param act: The activation function.
        :param drop_prob: The probability of DropPath.
        """
        super(FFN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
