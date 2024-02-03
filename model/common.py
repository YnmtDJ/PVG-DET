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
    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.1):
        """
        :param in_features: The number of input channels.
        :param hidden_features: The number of hidden channels.
        :param out_features: The number of output channels.
        :param drop_prob: The probability of DropPath.
        """
        super(FFN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.GroupNorm(32, hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.GroupNorm(32, out_features),
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
    Basic Convolution Block, including Convolution, GroupNorm, and activation function.
    """
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: The number of input channels.
        :param out_ch: The number of output channels.
        """
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, groups=4),
            nn.GroupNorm(32, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
