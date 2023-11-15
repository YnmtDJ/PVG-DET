from torch import nn


"""
Vision GNN: An Image is Worth Graph of Nodes
https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
"""


class Stem(nn.Module):
    """
    Image to Visual Word Embedding Overlap
    https://github.com/whai362/PVT
    """
    def __init__(self, in_ch=3, out_ch=768, act=nn.ReLU()):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param act: The activation function.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch//8),
            act,
            nn.Conv2d(out_ch//8, out_ch//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch//4),
            act,
            nn.Conv2d(out_ch//4, out_ch//2, 3, stride=2, padding=1),
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


