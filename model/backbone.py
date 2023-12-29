from torch import nn


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
