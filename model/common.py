from timm.layers import DropPath
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        :param input_dim: The input dimension.
        :param hidden_dim: The hidden dimension.
        :param output_dim: The output dimension.
        :param num_layers: The number of layers.
        """
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.Sequential(*[
            nn.Linear(in_ch, out_ch) for in_ch, out_ch in zip([input_dim] + h_dims, h_dims + [output_dim])
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
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


class BasicConv(nn.Module):
    """
    Basic Convolution Block, including Convolution, BatchNorm2d, and activation function.
    """
    def __init__(self, in_ch, out_ch, act=nn.GELU()):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        :param act: The activation function.
        """
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, groups=4),
            nn.BatchNorm2d(out_ch),
            act,
        )

    def forward(self, x):
        return self.conv(x)
